"""Merge a LoRA checkpoint into its base model and save full bf16 weights, for fast vLLM evaluation.

    python -m cues merge --adapter $RUNS/rl/olmo32b/checkpoint-300 --out $RUNS/eval/merged/olmo32b_ck300

Serving an unmerged adapter from vLLM is 20-30x slower than the base model on a 32B. The base defaults
to the adapter's own base_model_name_or_path (a registry key is accepted for --model). Runs on CPU (a
32B needs ~70 GB of RAM). The tokenizer and generation config are the base model's, so the merged model
differs from the base only in weights. Written to a private <out>.tmp<pid> and renamed, so a merge cut
short never looks finished; if <out> already holds a merged model, nothing is done.

Provenance: avdravid/reasoning_registers_grpo GRPO_clean scripts/merge.py.
"""
import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="base model (registry key or Hub id); default: the adapter's base_model_name_or_path")
    a = ap.parse_args()

    # Keep merging on CPU and prevent DeepSpeed from probing for a CUDA toolkit.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not os.environ.get("CUDA_HOME"):
        cu = glob.glob(os.path.join(sys.prefix, "lib", "python3*", "site-packages", "nvidia", "cu13"))
        if cu:
            os.environ["CUDA_HOME"] = cu[0]
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")

    from cues import models as M
    revision = None
    if a.model:
        entry = M.resolve(a.model)
        base, revision = entry["hub"], entry["revision"]
    else:
        base = json.load(open(Path(a.adapter) / "adapter_config.json"))["base_model_name_or_path"]
    out = Path(a.out)
    if (out / "config.json").exists():
        print(f"{out} already holds a merged model; nothing to do", flush=True)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    print(f"merging {a.adapter} into {base}@{revision} -> {out}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(tmp, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, revision=revision).save_pretrained(tmp)
    if (out / "config.json").exists():               # another merge of this checkpoint finished first
        shutil.rmtree(tmp)
        print(f"{out} was completed by another merge; kept that one", flush=True)
    else:
        shutil.rmtree(out, ignore_errors=True)       # anything there without config.json is incomplete
        tmp.rename(out)
        print("merged ->", out, flush=True)


if __name__ == "__main__":
    main()
