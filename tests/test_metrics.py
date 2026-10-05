from cues.metrics import cell_scores, pass_at_k, scores_from_records


def test_pass_at_k():
    assert pass_at_k(4, 0, 1) == 0.0
    assert pass_at_k(4, 4, 1) == 1.0
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-12
    assert pass_at_k(4, 1, 4) == 1.0          # n - c < k
    assert pass_at_k(2, 1, 4) is None          # fewer rollouts than k
    assert abs(pass_at_k(16, 8, 8) - (1 - 1 / 12870)) < 1e-12   # C(8,8)/C(16,8)


def test_cell_scores_deterministic():
    cell = {f"p{i}": [1, 0, 1, 0, 1, 0, 1, 1] * 2 for i in range(20)}
    a, b = cell_scores(cell), cell_scores(cell)
    assert a == b
    assert abs(a["pass@1"] - 0.625) < 1e-12
    assert a["pass@8"] == 1.0
    assert a["pass@16"] == 1.0
    assert len(a["pass@1_by_rollout_index"]) == 16
    assert a["pass@1_ci95"][0] <= a["pass@1"] <= a["pass@1_ci95"][1]


def test_scores_from_records():
    recs = [{"problem_key": "a", "rollout_index": i, "correct": i % 2 == 0, "hit_token_cap": False,
             "committed": True, "output_tokens": 10 + i} for i in range(8)]
    s = scores_from_records(recs, ks=(8,))
    assert s["pass@1"] == 0.5 and s["cap_rate"] == 0.0 and s["committed"] == 1.0
    assert s["n_problems"] == 1 and s["n_rollouts"] == 8
