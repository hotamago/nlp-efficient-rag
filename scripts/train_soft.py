"""Train the encode-once soft compressor (encoder + decoder LoRA, memory slots, projection).

uv run python scripts/train_soft.py --steps 10            # smoke test: speed + VRAM
uv run python scripts/train_soft.py --out data/soft/run1   # full run (1 epoch over train.jsonl)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch

from effrag.config import DATA_DIR, SEED, limit_gpu_memory
from effrag.soft.model import SoftConfig, SoftRAG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "soft" / "train.jsonl"))
    ap.add_argument("--out", default=str(DATA_DIR / "soft" / "run1"))
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (default: 1 epoch)")
    ap.add_argument("--accum", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--n-mem", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    data = [json.loads(x) for x in open(args.data)]
    rng.shuffle(data)
    steps = args.steps or len(data) // args.accum

    limit_gpu_memory()
    model = SoftRAG(SoftConfig(n_mem=args.n_mem))
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    params = model.trainable_parameters()
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, steps - args.warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    log = open(f"{args.out}.log", "a")
    t0, running, i = time.perf_counter(), 0.0, 0
    for step in range(steps):
        for _ in range(args.accum):
            ex = data[i % len(data)]
            i += 1
            docs = [tuple(d) for d in ex["docs"]]
            open_mask = [j == ex["open"] for j in range(len(docs))]
            loss = model.loss(ex["question"], docs, ex["answers"][0], open_mask) / args.accum
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.log_every == 0 or step == 0:
            el = time.perf_counter() - t0
            msg = {
                "step": step + 1, "loss": round(running / (1 if step == 0 else args.log_every), 4),
                "lr": sched.get_last_lr()[0], "s_per_step": round(el / (step + 1), 2),
                "eta_min": round(el / (step + 1) * (steps - step - 1) / 60, 1),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }  # fmt: skip
            print(json.dumps(msg), flush=True)
            log.write(json.dumps(msg) + "\n")
            log.flush()
            running = 0.0
        if (step + 1) % args.save_every == 0:
            model.save(args.out)
    model.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
