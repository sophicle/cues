"""GRPO on the MATH train set with the truncation penalty, for Olmo-3 and Qwen3 base models (paper §2.3, App. D).

    python -m cues train --model olmo7b --out $RUNS/rl/olmo7b                       # 300 steps, no cue
    python -m cues train --model olmo7b --out $RUNS/rl/olmo7b_cue --cue auto        # the cue forced on every prompt
    python -m cues train --model qwen14b --out $RUNS/rl/qwen14b --per-device 1 --gpu-mem 0.45
    python -m cues train --model qwen4b --out $RUNS/rl/x --smoke                    # 5 steps x 4 prompts
    python -m cues train --model olmo7b --out $RUNS/rl/olmo7b --resume              # the next allocation
Multi-GPU runs go through scripts/train.sh (accelerate launch); the 32B recipe is in the README.

Fixed by design: binary correctness reward with the truncation penalty (a rollout that hits the cap
scores 0 and is trained against; --no-trunc-penalty drops it instead); the full MATH train pool
(data/math_train.jsonl, 7,495 problems, no difficulty filter); 32 prompts x 16 samples = 512 rollouts
per step; temperature 1.0; on-policy (one optimizer step per generation batch); no KL; advantages
centred by the group mean, not scaled; token-level (DAPO) loss normalised over the generation batch;
LoRA r=64 alpha=128 on every attention and MLP projection; lr 1e-5 after 5 warm-up steps; bf16 base;
seed 20260813. Everything else is a flag with the paper's value as its default.

The cue is appended to the prompt as raw text, as in evaluation, and the policy is trained on the
completion only: the cued run learns to reason from the cue, the plain run has to find it.

Provenance: avdravid/reasoning_registers_grpo GRPO_clean src/train.py (65547cb).
"""
import argparse
import json
import os
import time
from pathlib import Path

from cues import models as M
from cues.prompts import STOPS, render
from cues.rl.reward import make_reward

HERE = Path(__file__).resolve().parent
POOL = HERE.parent.parent / "data" / "math_train.jsonl"


def build_dataset(path, prompt, cue, n=None):
    from datasets import Dataset
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if n:
        import random
        random.Random(0).shuffle(rows)
        rows = rows[:n]
    return Dataset.from_list([{"prompt": render(prompt, r["problem"], cue), "gt": r["gt"]} for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry key (olmo7b, olmo32b, qwen4b, qwen14b) or a Hub id with --family")
    ap.add_argument("--family", choices=["olmo", "qwen"], help="for a Hub id not in the registry")
    ap.add_argument("--prompt", choices=["boxed", "rlzero"], help="default: the registry's (rlzero for Olmo, boxed for Qwen)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems", default=str(POOL))
    ap.add_argument("--cue", default="none", help="none | auto (the registry's cue under the training prompt) | a literal opening")
    ap.add_argument("--no-trunc-penalty", action="store_true",
                    help="drop a capped rollout from the loss and the group mean instead of scoring it 0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=16)
    ap.add_argument("--max-completion", type=int, default=4096)
    ap.add_argument("--max-prompt", type=int, default=1024, help="only sets the vLLM context (prompt + completion); prompts are not cut")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--per-device", type=int, default=2, help="training microbatch per GPU")
    ap.add_argument("--gpu-mem", type=float, default=0.30, help="vLLM's share of each GPU when colocated (0.45 with sleep mode on a 14B)")
    ap.add_argument("--no-sleep", action="store_true", help="keep the vLLM engine resident during the backward (180 GB cards)")
    ap.add_argument("--vllm-tp", type=int, default=1, help="tensor-parallel size of the colocated vLLM engine; 8 shares one copy across a node (32B)")
    ap.add_argument("--zero3", action="store_true", help="shard the frozen base with DeepSpeed ZeRO-3 (cues/rl/zero3.json) instead of one copy per GPU (32B)")
    ap.add_argument("--no-liger", action="store_true", help="plain loss instead of the fused Liger loss")
    ap.add_argument("--no-save-text", action="store_true", help="skip the full-text rollout log")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
    ap.add_argument("--smoke", action="store_true", help="5 steps x 4 prompts at the real cap, into <out>_smoke")
    args = ap.parse_args()

    entry = M.resolve(args.model, args.family)
    if not entry["trainable"]:
        raise SystemExit(f"{entry['key']} is not an RL model of this repo (evaluate and search it; train olmo7b, olmo32b, qwen4b, qwen14b)")
    fam = M.FAMILIES[entry["family"]]
    prompt = args.prompt or entry["prompt"]
    cue = M.parse_cue(args.cue, entry, prompt)
    penalty = not args.no_trunc_penalty
    if args.smoke:
        args.steps, args.prompts_per_step, args.save_steps = 5, 4, 1000
        args.out += "_smoke"

    # Resuming a finished run would otherwise train one extra step.
    out = Path(args.out)
    if not args.smoke and (out / "DONE").exists():
        print(f"{out}/DONE exists: this run is finished (delete the file to train further)", flush=True)
        return
    if args.resume and not args.smoke:
        last = 0
        for c in out.glob("checkpoint-*"):
            if c.name.split("-", 1)[1].isdigit():
                try:
                    last = max(last, json.load(open(c / "trainer_state.json"))["global_step"])
                except (OSError, ValueError, KeyError):
                    pass
        if last >= args.steps:
            (out / "DONE").touch()
            print(f"newest checkpoint is at step {last} >= --steps {args.steps}: marked DONE, nothing to do", flush=True)
            return

    import logging
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from cues.rl import patches
    patches.apply(liger=not args.no_liger)
    if not penalty:
        # a dropped rollout is a None reward, which TRL reports every step by printing the whole completion
        logging.getLogger("trl.trainer.grpo_trainer").addFilter(
            lambda rec: not rec.getMessage().startswith("All reward functions returned None"))
    if args.zero3:
        patches.apply_zero3_sync()

    world = int(os.environ.get("WORLD_SIZE") or torch.cuda.device_count())
    assert world % args.vllm_tp == 0, f"--vllm-tp {args.vllm_tp} must divide the world size {world}"
    rollouts = args.prompts_per_step * args.num_generations
    accum = max(1, round(rollouts / (args.per_device * world)))
    glob = args.per_device * world * accum
    assert glob % args.num_generations == 0, f"global batch {glob} not divisible by G={args.num_generations}"
    ds = build_dataset(args.problems, prompt, cue, n=20 if args.smoke else None)
    print(f"{len(ds)} prompts | {entry['key']} = {entry['hub']}@{entry['revision']} | prompt={prompt} cue={cue!r} "
          f"trunc_penalty={penalty} | zero3={args.zero3} vllm_tp={args.vllm_tp} liger={not args.no_liger} | "
          f"per_device {args.per_device} x world {world} x accum {accum} = {glob} rollouts/step "
          f"({glob // args.num_generations} prompts x {args.num_generations})", flush=True)

    # Qwen uses <|im_end|> as EOS and <|endoftext|> as padding; TRL accepts both endings.
    tok = AutoTokenizer.from_pretrained(entry["hub"], revision=entry["revision"], **({"eos_token": fam["eos"]} if fam["eos"] else {}))
    eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None}
    gen_kwargs = {"stop": STOPS[prompt]}
    if fam["stop_tokens"]:
        gen_kwargs["stop_token_ids"] = [tok.convert_tokens_to_ids(t) for t in fam["stop_tokens"]]
    print(f"eos={tok.eos_token!r}({tok.eos_token_id}) pad={tok.pad_token!r}({tok.pad_token_id}) gen_kwargs={gen_kwargs}", flush=True)

    cfg = GRPOConfig(
        output_dir=args.out,
        model_init_kwargs={"dtype": torch.bfloat16, **({"revision": entry["revision"]} if entry["revision"] else {})},
        per_device_train_batch_size=args.per_device,
        gradient_accumulation_steps=accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        vllm_max_model_length=args.max_prompt + args.max_completion,
        temperature=args.temperature,
        generation_kwargs=gen_kwargs,
        beta=0.0,
        num_iterations=1,
        epsilon_high=0.28,                      # inert at one iteration; DAPO's value if it ever rises
        loss_type="dapo",
        scale_rewards="none",
        # penalty: the capped rollout scores 0 in its group AND gets a gradient. no penalty: it is
        # dropped from both (its reward is None; TRL leaves it out of the baseline and masks its loss).
        mask_truncated_completions=not penalty,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        max_steps=args.steps,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=None,                  # keep every checkpoint: the trajectory is the experiment
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True} if args.zero3 else None,   # PEFT + ZeRO-3 needs it
        use_liger_kernel=not args.no_liger,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=args.vllm_tp,
        deepspeed=str(HERE / "zero3.json") if args.zero3 else None,
        vllm_gpu_memory_utilization=args.gpu_mem,
        vllm_enable_sleep_mode=not args.no_sleep,
        reward_weights=[1.0],
        report_to=[],
        seed=20260813,
    )
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    class StepEnd(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if state.is_world_process_zero:
                print(f"[phase] step {state.global_step} train_end {time.time():.1f}", flush=True)

    class KeepSaveSteps(TrainerCallback):
        """A resumed TrainerState carries the save_steps the checkpoint was written with and HF does
        not re-apply the flag; put it back so the cadence survives a resume."""
        def on_step_begin(self, a, state, control, **kw):
            state.save_steps = args.save_steps
            return control

    trainer = GRPOTrainer(
        model=entry["hub"],
        reward_funcs=[make_reward(eos_ids, args.max_completion, prompt, out / "rollout_log",
                                  save_text=not args.no_save_text, penalty=penalty)],
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[StepEnd(), KeepSaveSteps()],
    )
    out.mkdir(parents=True, exist_ok=True)
    json.dump({**vars(args), "hub": entry["hub"], "revision": entry["revision"], "family": entry["family"],
               "prompt": prompt, "cue_text": cue, "trunc_penalty": penalty}, open(out / "run_args.json", "w"), indent=1)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(args.out)
    if not args.smoke:
        (out / "DONE").touch()
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
