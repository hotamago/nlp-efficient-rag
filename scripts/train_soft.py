"""Train the encode-once soft compressor (encoder + decoder LoRA, memory slots, projection).

uv run python scripts/train_soft.py --steps 10            # smoke test: speed + VRAM
uv run python scripts/train_soft.py --out data/soft/run1   # full run (1 epoch over train.jsonl)

Every --ckpt-every steps the full training state (adapters, projection, optimizer, scheduler, data
position, RNG) is written atomically to <out>/ckpt; rerunning the same command resumes from it.
--start-step N resumes from weights saved in <out> without optimizer state (older runs).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

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
    ap.add_argument("--ckpt-every", type=int, default=100, help="resumable checkpoint interval")
    ap.add_argument("--start-step", type=int, default=0, help="resume from weights in --out only")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    data = [json.loads(x) for x in open(args.data)]
    rng.shuffle(data)
    steps = args.steps or len(data) // args.accum

    out = Path(args.out)
    ckpt = out / "ckpt"
    if not (ckpt / "trainer.pt").exists() and (out / "ckpt.old" / "trainer.pt").exists():
        ckpt = out / "ckpt.old"  # interrupted between the two renames in save_ckpt()
    limit_gpu_memory()
    if (ckpt / "trainer.pt").exists():
        model = SoftRAG.load(ckpt)
    elif args.start_step:
        model = SoftRAG.load(out)
    else:
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
    start = args.start_step
    if (ckpt / "trainer.pt").exists():
        st = torch.load(ckpt / "trainer.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng_cpu"])
        if st["rng_cuda"] is not None:
            torch.cuda.set_rng_state(st["rng_cuda"])
        start = st["step"]
    else:
        for _ in range(start):  # weights-only resume: fresh Adam moments, schedule at step N
            sched.step()
    if start:
        print(f"resumed at step {start}/{steps} (lr {sched.get_last_lr()[0]:.2e})", flush=True)

    def save_ckpt(step: int) -> None:
        """Write to ckpt.tmp, then swap, so a crash mid-write never loses the last checkpoint."""
        tmp, cur, old = out / "ckpt.tmp", out / "ckpt", out / "ckpt.old"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save(tmp)
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None},
                   tmp / "trainer.pt")  # fmt: skip
        shutil.rmtree(old, ignore_errors=True)
        if cur.exists():
            cur.rename(old)
        tmp.rename(cur)
        shutil.rmtree(old, ignore_errors=True)

    model.train()
    log = open(f"{args.out}.log", "a")
    t0, running, i = time.perf_counter(), 0.0, start * args.accum
    for step in range(start, steps):
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
            el, done = time.perf_counter() - t0, step + 1 - start
            msg = {
                "step": step + 1, "loss": round(running / (1 if step == 0 else args.log_every), 4),
                "lr": sched.get_last_lr()[0], "s_per_step": round(el / done, 2),
                "eta_min": round(el / done * (steps - step - 1) / 60, 1),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }  # fmt: skip
            print(json.dumps(msg), flush=True)
            log.write(json.dumps(msg) + "\n")
            log.flush()
            running = 0.0
        if (step + 1) % args.save_every == 0:
            model.save(args.out)
        if (step + 1) % args.ckpt_every == 0:
            save_ckpt(step + 1)
    model.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
