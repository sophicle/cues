"""The paper's graders: extract the committed answer, check equivalence with math-verify.

Evaluation and the cue search grade exactly as the paper did (sophicle/reason, data/bank/grader.py for the boxed
prompt and data/bank_rlzero/grader.py for RL-Zero; the RL-Zero file's SHA-256 begins f59837872a05):

  boxed    the last brace-balanced \\boxed{...}; when there is none, the text after the last "Final answer"
  rlzero   the last line that starts with "Answer:" (optional $ around the value), else as boxed

is_correct returns True or False when the committed answer and the reference both parse, and None otherwise (no
answer committed, or either side unparseable), as the paper's graders do. pass@1 counts None as not correct.
Both sides are wrapped in \\boxed{} before math_verify.parse, which lets it read fractions, tuples and \\text{}
forms a bare parse drops without loosening the equality test. HumanEval is graded by execution (cues.exec_grade).

The RL reward keeps the rule the paper's Olmo-3-7B and Qwen3 RL runs were trained with (reward_is_correct below;
avdravid/reasoning_registers_grpo src/math_grade.py and lenient_correct in src/grpo_olmo.py, the same on every
training branch): rlzero takes the last "Answer:" line (a bold **Answer:** too, trailing period and $...$ stripped),
else the last box, else a "Final answer:" line; boxed takes the last box, else an "answer: ..." / "final answer = ..."
capture. Anything uncommitted or unparseable scores False. Changing the reward would change the training runs, so
it is kept apart from the evaluation graders.
"""
import re

from math_verify import parse, verify

_ANSWER_LINE = re.compile(r"(?:^|\n)\s*Answer:\s*\$?([^\n$]+?)\s*\$?\s*(?:\n|$)")


def boxed(text):
    """Last \\boxed{...} with balanced braces, else a trailing 'Final answer:' capture, else None. Scans backwards
    from the last \\boxed{ so a completion that boxes a first attempt, revises, and boxes again returns the final
    answer, and nested LaTeX like \\boxed{\\frac{\\sqrt{2}}{2}} survives."""
    if not text:
        return None
    start = text.rfind("\\boxed{")
    while start != -1:
        i = start + len("\\boxed{")
        depth = 1
        while i < len(text) and depth:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if not depth:
                    return text[start + len("\\boxed{"):i]
            i += 1
        start = text.rfind("\\boxed{", 0, start)      # unbalanced: try an earlier one
    t = re.findall(r"[Ff]inal answer:?\s*(.+)", text)
    return t[-1].strip() if t else None


def extract(text, style="boxed"):
    """The answer the text commits to under the prompt's contract, or None."""
    if style == "rlzero":
        m = _ANSWER_LINE.findall(text or "")
        if m:
            return m[-1].strip()
    return boxed(text)


def _parse(s):
    try:
        return parse("\\boxed{" + str(s) + "}")
    except Exception:
        return None


def is_correct(gt, text, style="boxed"):
    """True/False when gradable; None when no answer is committed or either side will not parse."""
    a = extract(text, style)
    if a is None:
        return None
    g, p = _parse(gt), _parse(a)
    if not g or not p:
        return None
    try:
        return bool(verify(g, p))
    except Exception:
        return None


# ---- the RL reward's rule, unchanged from the paper's RL runs (math_grade.py + grpo_olmo.lenient_correct) ----------
_REWARD_ANSWER_LINE = re.compile(r"^[ \t]*\**\s*Answer\**\s*:\s*\**\s*(.+?)\s*$", re.M)
_REWARD_ANS = re.compile(r"(?:final answer|answer)\s*[:=]\s*\$?([^\n$]+)", re.I)


def _reward_boxed(text):
    """math_grade.boxed: the box that ends the text, else the last box with at most one level of nested braces,
    else a trailing 'Final answer:' capture."""
    b = re.findall(r"\\boxed\{(.+?)\}\s*$", text, re.S)
    if not b:
        b = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if b:
        return b[-1]
    t = re.findall(r"[Ff]inal answer:?\s*(.+)", text)
    return t[-1].strip() if t else None


def reward_extract(text, style="boxed"):
    """math_grade.extract: under rlzero the last Answer: line, else _reward_boxed; under boxed, _reward_boxed."""
    text = text or ""
    if style == "rlzero":
        m = _REWARD_ANSWER_LINE.findall(text)
        if m:
            s = m[-1].strip().rstrip(".")
            while len(s) > 1 and s[0] == "$" and s[-1] == "$":
                s = s[1:-1].strip()
            return s
    return _reward_boxed(text)


def _reward_verify(gt, a):
    if a is None:
        return False
    g, p = _parse(gt), _parse(a)
    if not g or not p:
        return False
    try:
        return bool(verify(g, p))
    except Exception:
        return False


def reward_is_correct(gt, text, style="boxed"):
    """grpo_olmo.lenient_correct: True only for a committed, parseable, equivalent answer."""
    text = text or ""
    if style == "rlzero":
        return _reward_verify(gt, reward_extract(text, style))
    if _reward_boxed(text) is not None:
        return _reward_verify(gt, _reward_boxed(text))
    m = _REWARD_ANS.findall(text)
    return bool(m) and _reward_verify(gt, m[-1].strip())
