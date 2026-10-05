"""The model registry: what differs between models, in one place.

Each entry names the Hub id and the commit the paper pinned, the family (which decides the
end-of-rollout token and stop tokens), the prompt the model is trained and evaluated under, and the
cue the label-free search selected for it under each prompt (paper Table 1 / App. A, Table 1).

Any Hub id or local directory is accepted wherever a registry key is; pass --family for it.

Provenance: sophicle/reason registers/nominees.py (cues), scripts/eval/grid_shards.py (pins),
avdravid/reasoning_registers_grpo GRPO_clean src/families.py (families).
"""
import hashlib
import re

# family -> how a rollout ends. Qwen3 base ships a tokenizer whose EOS is <|endoftext|> while the
# model closes a turn with <|im_end|>; the trainer overrides EOS so a finished rollout is not
# mistaken for a truncated one, and generation stops on both ids.
FAMILIES = {
    "olmo": {"eos": None, "stop_tokens": []},
    "qwen": {"eos": "<|im_end|>", "stop_tokens": ["<|im_end|>", "<|endoftext|>"]},
}

MODELS = {
    "olmo7b": {"hub": "allenai/Olmo-3-1025-7B", "revision": "a81bae42db3975be1671e27b9c9a56da1a9f980f",
               "family": "olmo", "prompt": "rlzero", "trainable": True,
               "cues": {"rlzero": ".\n\nOkay", "boxed": " \n\nOkay"}},
    "olmo32b": {"hub": "allenai/Olmo-3-1125-32B", "revision": "c2b61dae89a1ad10e4ad5653d0e46b590902607b",
                "family": "olmo", "prompt": "rlzero", "trainable": True,
                "cues": {"rlzero": ".\n\nOkay", "boxed": " \n\n"}},
    "qwen4b": {"hub": "Qwen/Qwen3-4B-Base", "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
               "family": "qwen", "prompt": "boxed", "trainable": True,
               "cues": {"boxed": " To determine", "rlzero": ".\nTo"}},
    "qwen14b": {"hub": "Qwen/Qwen3-14B-Base", "revision": "0b0bd3732e2c374d483664439ea334928b65f304",
                "family": "qwen", "prompt": "boxed", "trainable": True,
                "cues": {"boxed": " Alright,", "rlzero": ".\n\nTo"}},
    "olmo7b_rlzero": {"hub": "allenai/Olmo-3-7B-RL-Zero-Math", "revision": "dee26a4074b88215cd1326472e1ab14d0259d892",
                      "family": "olmo", "prompt": "rlzero", "trainable": False,
                      "cues": {"rlzero": ".\n\nOkay", "boxed": " \n\nOkay"}},
    "olmo7b_instruct": {"hub": "allenai/Olmo-3-7B-Instruct", "revision": None,
                        "family": "olmo", "prompt": "chat", "chat": True, "trainable": False, "cues": {}},
    "olmo7b_think": {"hub": "allenai/Olmo-3-7B-Think", "revision": None,
                     "family": "olmo", "prompt": "chat", "chat": True, "trainable": False, "cues": {}},
}


def guess_family(name):
    n = name.lower()
    if "qwen" in n:
        return "qwen"
    return "olmo"


def resolve(name, family=None):
    """A registry entry for a key, or a synthesized one for a Hub id / local path."""
    if name in MODELS:
        e = dict(MODELS[name])
        if family and family != e["family"]:
            raise SystemExit(f"{name} is family {e['family']!r}, not {family!r}")
        e["key"] = name
        return e
    fam = family or guess_family(name)
    if fam not in FAMILIES:
        raise SystemExit(f"unknown family {fam!r}; one of {sorted(FAMILIES)}")
    return {"key": re.sub(r"[^A-Za-z0-9._-]+", "_", name.rstrip("/").split("/")[-1]),
            "hub": name, "revision": None, "family": fam,
            "prompt": "boxed" if fam == "qwen" else "rlzero", "trainable": fam in ("olmo", "qwen"),
            "cues": {}}


def cue_for(entry, prompt):
    """The selected cue of a model under a prompt (what --cue auto means)."""
    try:
        return entry["cues"][prompt]
    except KeyError:
        raise SystemExit(f"no selected cue for {entry['key']} under the {prompt} prompt; "
                         f"run `python -m cues search` or pass the cue string literally")


def cue_slug(cue):
    """Readable directory name plus a hash of the exact cue; empty cue stays `none`."""
    if cue == "":
        return "none"
    s = cue
    for a, b in ((".\n\n\n", "p3"), (".\n\n", "p2"), (".\n", "p1"), ("\n\n\n", "n3"), ("\n\n", "n2"), ("\n", "n1"),
                 (" ", "sp"), ("(", "paren"), (")", "cparen"), ("'", "aps"), ("$", "dollar"), (",", "c"),
                 (".", "dot"), (":", "colon")):
        s = s.replace(a, b)
    readable = (re.sub(r"[^A-Za-z0-9]", "x", s) or "empty")[:64]
    return readable + "-" + hashlib.sha256(cue.encode("utf-8")).hexdigest()


def parse_cue(spec, entry, prompt):
    """`--cue`/`--cues` value -> the literal string: 'none' -> '', 'auto' -> the registry cue, else itself."""
    if spec == "none":
        return ""
    if spec == "auto":
        return cue_for(entry, prompt)
    if "\\n" in spec:
        raise SystemExit(f"cue {spec!r} contains a literal backslash-n; in bash write $'.\\n\\nOkay' so the newlines are real")
    return spec
