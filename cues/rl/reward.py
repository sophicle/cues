"""The reward: 1 if the final answer is correct, 0 otherwise; a rollout that hit the cap scores 0.

A rollout that reaches the completion cap never finished, so under the truncation penalty it scores 0:
it votes in its group mean like any other wrong rollout AND it receives a gradient (the trainer leaves
it in the loss mask). Non-termination is treated as an ordinary wrong answer and trained against; it is
length pressure, and under a token-level loss a rollout at the cap carries more gradient weight than a
short finished one, so watch `completions/mean_terminated_length` in the log.

With penalty=False the capped rollout's reward is None: TRL leaves it out of the group baseline and,
with mask_truncated_completions=True, out of the loss (neither vote nor gradient). TRL's own truncation
test is "the last token is not EOS", so in that mode a rollout that ended on a stop string is masked too.

Grading is cues.grading.reward_is_correct under the training prompt's contract (an "Answer:" line for rlzero,
the last box otherwise); an uncommitted answer scores 0. Every rollout is logged: one JSONL record per
rollout per rank (reward, capped, length, opener) and the full text in a gzip stream beside it; ranks
write separate files. `python -m cues openers <run>` reads them.

Provenance: avdravid/reasoning_registers_grpo GRPO_clean src/reward.py.
"""
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

from cues.grading import reward_is_correct as is_correct   # the reward keeps the paper's RL rule; see cues/grading.py
from cues.prompts import GRADING

_OPENER = re.compile(r"\s*(\S+)")
_REGISTER = re.compile(r"^\s*(Okay|Alright|Hmm)\b", re.I)


def opener(text, prompt):
    """First word of the solution. Under rlzero the prompt ends mid-sentence, so a base model often
    completes that sentence first; skip to the next line unless the text already opens in register."""
    if prompt == "rlzero" and not _REGISTER.match(text):
        i = text[:200].find("\n")
        if i != -1:
            text = text[i + 1:]
    m = _OPENER.match(text)
    return m.group(1) if m else ""


def make_reward(eos_ids, max_completion, prompt, log_dir, save_text=True, penalty=True):
    rank = os.environ.get("RANK", "0")
    style = GRADING[prompt]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    rec_path = log_dir / f"rollouts_rank{rank}.jsonl"
    text_path = log_dir / f"text_rank{rank}.jsonl.gz"
    state = {"step": 0}

    def correct(completions, gt, completion_ids, prompts=None, trainer_state=None, **_):
        state["step"] += 1
        gstep = getattr(trainer_state, "global_step", None)
        if rank == "0":
            print(f"[phase] gen_end {time.time():.1f}", flush=True)
        rewards, recs, texts = [], [], []
        for i, (text, g, ids) in enumerate(zip(completions, gt, completion_ids)):
            # TRL's own test for "did not finish": last token is not EOS/pad. Combined with the length,
            # that separates a rollout that hit the cap from one that ended on a stop string.
            unfinished = len(ids) == 0 or ids[-1] not in eos_ids
            capped = unfinished and len(ids) >= max_completion
            if capped:
                r = 0.0 if penalty else None
            else:
                r = float(is_correct(g, text, style) or 0.0)
            rewards.append(r)
            pid = hashlib.blake2s(prompts[i].encode(), digest_size=6).hexdigest() if prompts is not None else None
            op = opener(text, prompt)
            recs.append({"gstep": gstep, "step": state["step"], "pid": pid, "reward": r, "capped": capped,
                         "n_tokens": len(ids), "opener": op, "register": bool(_REGISTER.match(op))})
            if save_text:
                texts.append({"gstep": gstep, "pid": pid, "reward": r, "capped": capped, "n_tokens": len(ids), "text": text})
        with open(rec_path, "a") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        if texts:
            with gzip.open(text_path, "at") as f:
                for t in texts:
                    f.write(json.dumps(t) + "\n")
        if rank == "0":
            n = len(recs)
            scored = [r for r in rewards if r is not None]
            top = {}
            for rec in recs:
                top.setdefault(rec["opener"], []).append(rec["reward"] or 0.0)
            shares = sorted(((len(v) / n, k, sum(v) / len(v)) for k, v in top.items()), reverse=True)[:4]
            print(f"[step {state['step']:>4}] reward {sum(scored) / max(1, len(scored)):.3f}  "
                  f"capped {sum(r['capped'] for r in recs) / n:.3f}  register {sum(r['register'] for r in recs) / n:.3f}  "
                  f"tokens {sum(r['n_tokens'] for r in recs) / n:.0f}  openers "
                  + "  ".join(f"{k!r} {s:.2f} (r={m:.2f})" for s, k, m in shares), flush=True)
        return rewards

    correct.__name__ = "correct"
    return correct
