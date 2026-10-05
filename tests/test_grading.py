import pytest

from cues.grading import boxed, extract, is_correct


def test_last_box_wins():
    t = "First guess \\boxed{5}. Wait, recheck: the answer is \\boxed{7}"
    assert boxed(t) == "7"


def test_nested_braces_survive():
    assert boxed("so \\boxed{\\frac{\\sqrt{2}}{2}}.") == "\\frac{\\sqrt{2}}{2}"


def test_multiple_boxes_ending_in_brace():
    t = "\\boxed{5}. Later \\boxed{\\frac{1}{2}}"
    assert boxed(t) == "\\frac{1}{2}"


def test_no_box_is_none():
    assert boxed("no answer here") is None
    assert extract("no answer here", "rlzero") is None


def test_rlzero_answer_line_preferred_over_box():
    t = "we get \\boxed{3}\n\nAnswer: 4\n"
    assert extract(t, "rlzero") == "4"
    assert extract(t, "boxed") == "3"


def test_rlzero_answer_line_is_the_papers_rule():
    assert extract("...\nAnswer: $12$\n", "rlzero") == "12"          # optional dollars around the value
    assert extract("...\n**Answer:** 12\n", "rlzero") is None        # a bold label is not an Answer line in the paper's grader
    assert extract("Final answer: 7", "boxed") == "7"                 # the "Final answer" fallback, as the paper's graders have it
    assert is_correct("7", "Final answer: 7") is True


def test_reward_keeps_the_rl_rule():
    from cues.grading import reward_extract, reward_is_correct
    assert reward_extract("...\n**Answer:** $12$.\n", "rlzero") == "12"
    assert reward_is_correct("-3", "...\nAnswer: \\boxed{-3}", "rlzero") is True
    assert reward_is_correct("18", "Answer: 18", "rlzero") is True
    assert reward_is_correct("3, 5, 7", "Answer: \\boxed{3}, \\boxed{5}, \\boxed{7}", "rlzero") is True
    assert reward_is_correct("4", "so it is 4.\nFinal answer: 4", "rlzero") is True
    assert reward_is_correct("9", "the answer: 9", "boxed") is True


def test_rlzero_falls_back_to_box():
    assert extract("so \\boxed{9} done", "rlzero") == "9"


def test_is_correct_equivalence():
    assert is_correct("\\frac{1}{2}", "the answer is \\boxed{0.5}") is True
    assert is_correct("\\frac{1}{2}", "the answer is \\boxed{0.6}") is False
    assert is_correct("18", "Answer: 18", "rlzero") is True
    assert is_correct("C", "\\boxed{C}") is True


def test_uncommitted_is_none():
    assert is_correct("18", "I am not sure.") is None
    assert is_correct("18", "I am not sure.", "rlzero") is None


def test_tuple_answer():
    gt = "\\left( 3, \\frac{\\pi}{2} \\right)"
    assert is_correct(gt, "\\boxed{(3, \\frac{\\pi}{2})}") is True
    assert is_correct(gt, "\\boxed{(3, \\pi)}") is False
