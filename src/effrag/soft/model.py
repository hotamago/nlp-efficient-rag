"""Encode-once soft context compression (PISCO-style, Louis et al. 2025, arXiv 2501.16075).

One causal LM plays both roles through two LoRA adapters:
- encoder: reads one document (<= doc_max_tokens) followed by n_mem learned memory slots and
  returns the last hidden states of those slots -> n_mem vectors per document (query-independent,
  so a corpus can be compressed once offline);
- decoder: reads the Lost-in-the-Middle prompt in which each document's text is replaced by its
  memory vectors (the title stays as text) and generates the answer.

Hybrid mode keeps some documents (e.g. the reranker's top-1) as plain text and the rest as
memory - the RAG-level analogue of mixing fine-grained selected entries with compressed ones
(DeepSeek sparse attention / CSA). See docs/research_soft_compression.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from effrag.generation import CLOSEDBOOK_TEMPLATE, QA_TEMPLATE

MEM = "<<MEM>>"


@dataclass
class SoftConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    n_mem: int = 8
    doc_max_tokens: int = 128
    lora_r: int = 16
    lora_dropout: float = 0.05


def _set_active_adapter(model: nn.Module, name: str) -> None:
    """Switch the active LoRA adapter without touching requires_grad (peft's set_adapter freezes
    the inactive adapter, which would stop the encoder from training)."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    for m in model.modules():
        if isinstance(m, BaseTunerLayer):
            m._active_adapter = name


class SoftRAG(nn.Module):
    def __init__(self, cfg: SoftConfig, device: str = "cuda", adapters_dir: str | None = None):
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(cfg.base_model)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, dtype=dtype, attn_implementation="sdpa"
        )
        h = base.config.hidden_size
        if adapters_dir:
            dec = f"{adapters_dir}/decoder"
            self.lm = PeftModel.from_pretrained(base, dec, adapter_name="decoder")
            self.lm.load_adapter(f"{adapters_dir}/encoder", adapter_name="encoder")
        else:
            lcfg = LoraConfig(
                r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=cfg.lora_dropout,
                target_modules="all-linear", task_type="CAUSAL_LM",
            )  # fmt: skip
            self.lm = get_peft_model(base, lcfg, adapter_name="decoder")
            self.lm.add_adapter("encoder", lcfg)
        # memory slots and an identity-initialised projection from encoder to decoder space; the
        # output is L2-normalised and rescaled to the token-embedding norm (last hidden states are
        # ~600x larger than input embeddings in Qwen2.5)
        emb_w = base.get_input_embeddings().weight.float()
        std = emb_w.std().item()
        self.out_scale = nn.Parameter(torch.tensor(emb_w.norm(dim=-1).mean().item()))
        self.mem = nn.Parameter(torch.randn(cfg.n_mem, h) * std)
        self.proj = nn.Linear(h, h)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(h))
            self.proj.bias.zero_()
        if adapters_dir:
            extra = torch.load(
                f"{adapters_dir}/soft_extra.pt", map_location="cpu", weights_only=True
            )
            self.mem.data.copy_(extra["mem"])
            self.proj.load_state_dict(extra["proj"])
            self.out_scale.data.fill_(float(extra["out_scale"]))
        self.to(self.device)

    # ----------------------------------------------------------------- parameters
    def trainable_parameters(self) -> list[nn.Parameter]:
        params = [p for n, p in self.lm.named_parameters() if "lora_" in n]
        for p in params:
            p.requires_grad_(True)
        return params + [self.mem, self.out_scale, *self.proj.parameters()]

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(out)  # writes ./decoder (default) and ./encoder subfolders
        if (out / "adapter_config.json").exists():  # peft stores the first adapter at the root
            (out / "decoder").mkdir(exist_ok=True)
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                (out / f).replace(out / "decoder" / f)
        extra = {"mem": self.mem.detach().cpu(), "proj": self.proj.state_dict(),
                 "out_scale": self.out_scale.item()}  # fmt: skip
        torch.save(extra, out / "soft_extra.pt")
        (out / "soft_config.json").write_text(json.dumps(asdict(self.cfg)))

    @classmethod
    def load(cls, adapters_dir: str | Path, device: str = "cuda") -> SoftRAG:
        cfg = SoftConfig(**json.loads((Path(adapters_dir) / "soft_config.json").read_text()))
        return cls(cfg, device=device, adapters_dir=str(adapters_dir))

    # ----------------------------------------------------------------- encoder
    def _embed(self, ids: list[int]) -> torch.Tensor:
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.lm.get_input_embeddings()(t)

    def encode_docs(self, docs: list[tuple[str, str]], batch_size: int = 32) -> torch.Tensor:
        """(title, text) pairs -> memory vectors of shape (n_docs, n_mem, hidden)."""
        _set_active_adapter(self.lm, "encoder")
        emb_layer = self.lm.get_input_embeddings()
        dtype = emb_layer.weight.dtype
        outs = []
        for i in range(0, len(docs), batch_size):
            chunk = docs[i : i + batch_size]
            ids = [
                self.tok(f"(Title: {t}) {x}", add_special_tokens=False,
                         truncation=True, max_length=self.cfg.doc_max_tokens).input_ids
                for t, x in chunk
            ]  # fmt: skip
            L = max(len(x) for x in ids) + self.cfg.n_mem
            pad = self.tok.pad_token_id
            input_ids = torch.full((len(ids), L - self.cfg.n_mem), pad, device=self.device)
            mask = torch.zeros((len(ids), L), dtype=torch.long, device=self.device)
            for r, x in enumerate(ids):  # left padding so memory slots sit at the end
                input_ids[r, L - self.cfg.n_mem - len(x) :] = torch.tensor(x, device=self.device)
                mask[r, L - self.cfg.n_mem - len(x) :] = 1
            embeds = torch.cat(
                [emb_layer(input_ids), self.mem.to(dtype).expand(len(ids), -1, -1)], dim=1
            )
            pos = (mask.cumsum(1) - 1).clamp(min=0)
            hs = self.lm(inputs_embeds=embeds, attention_mask=mask, position_ids=pos,
                         output_hidden_states=True).hidden_states[-1]  # fmt: skip
            v = self.proj(hs[:, -self.cfg.n_mem :].float())
            outs.append(nn.functional.normalize(v, dim=-1) * self.out_scale)
        return torch.cat(outs)

    # ----------------------------------------------------------------- decoder
    def _prompt_pieces(self, question: str, docs: list[tuple[str, str]], open_mask: list[bool]):
        """Chat-formatted LITM prompt split around memory placeholders."""
        if docs:
            ctx = "\n".join(
                f"Document [{i}](Title: {t}) " + (x if o else MEM)
                for i, ((t, x), o) in enumerate(zip(docs, open_mask, strict=True), 1)
            )
            user = QA_TEMPLATE.format(context=ctx, question=question)
        else:
            user = CLOSEDBOOK_TEMPLATE.format(question=question)
        text = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=True)  # fmt: skip
        return text.split(MEM)

    def decoder_inputs(self, question, docs, mem: torch.Tensor | None, open_mask=None):
        """Build decoder input embeddings; mem holds vectors for the non-open docs, in order."""
        open_mask = open_mask or [False] * len(docs)
        pieces = self._prompt_pieces(question, docs, open_mask)
        dtype = self.lm.get_input_embeddings().weight.dtype
        parts = [self._embed(self.tok(pieces[0], add_special_tokens=False).input_ids)]
        for j, piece in enumerate(pieces[1:]):
            parts.append(mem[j].to(dtype))
            parts.append(self._embed(self.tok(piece, add_special_tokens=False).input_ids))
        return torch.cat(parts)

    def loss(self, question, docs, answer: str, open_mask=None) -> torch.Tensor:
        open_mask = open_mask or [False] * len(docs)
        closed = [d for d, o in zip(docs, open_mask, strict=True) if not o]
        mem = self.encode_docs(closed) if closed else None
        _set_active_adapter(self.lm, "decoder")
        prompt = self.decoder_inputs(question, docs, mem, open_mask)
        target = self.tok(answer + "<|im_end|>", add_special_tokens=False).input_ids
        embeds = torch.cat([prompt, self._embed(target)])[None]
        labels = torch.full(embeds.shape[:2], -100, dtype=torch.long, device=self.device)
        labels[0, -len(target) :] = torch.tensor(target, device=self.device)
        return self.lm(inputs_embeds=embeds, labels=labels).loss

    @torch.no_grad()
    def generate(self, question, docs, mem=None, open_mask=None, max_new_tokens: int = 48,
                 base: bool = False) -> str:  # fmt: skip
        """Greedy answer. Pass precomputed `mem` (encode-once) or let it encode the closed docs.
        base=True runs the untouched base model on plain text (all docs open) as a reference."""
        if base:
            with self.lm.disable_adapter():
                return self._generate(question, docs, None, [True] * len(docs), max_new_tokens)
        open_mask = open_mask or [False] * len(docs)
        return self._generate(question, docs, mem, open_mask, max_new_tokens)

    def _generate(self, question, docs, mem, open_mask, max_new_tokens) -> str:
        if mem is None and not all(open_mask):
            closed = [d for d, o in zip(docs, open_mask, strict=True) if not o]
            mem = self.encode_docs(closed) if closed else None
        _set_active_adapter(self.lm, "decoder")
        embeds = self.decoder_inputs(question, docs, mem, open_mask)[None]
        out = self.lm.generate(inputs_embeds=embeds, attention_mask=torch.ones(embeds.shape[:2],
                               dtype=torch.long, device=self.device), max_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=self.tok.pad_token_id)  # fmt: skip
        return self.tok.decode(out[0], skip_special_tokens=True).strip()
