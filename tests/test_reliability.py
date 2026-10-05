import json
import os
import subprocess
from pathlib import Path

import pytest

from cues.evaluate import cached_cell, model_identity
from cues.models import cue_slug
from cues import exec_grade


def test_distinct_cues_have_distinct_names():
    cues = ['', 'none', ' ', 'sp', ',', 'c', '\n', 'n1', '.\n\n', 'p2', '🐔', '?']
    assert len({cue_slug(c) for c in cues}) == len(cues)
    assert cue_slug('') == 'none'


def test_cache_requires_same_experiment_and_complete_indices(tmp_path):
    f = tmp_path / 'rollouts_p00000.jsonl'
    problems = [{'problem_key': 'a'}]
    config = {'seed': 1, 'max_tokens': 100}
    def write(records):
        f.write_text(''.join(json.dumps(r) + '\n' for r in records))
    record = {'problem_key': 'a', 'rollout_index': 0, 'experiment': config}
    write([record])
    assert cached_cell(tmp_path, f, config, problems, 0, 1)
    assert not cached_cell(tmp_path, f, config, problems, 0, 2)
    for changed in [{'seed': 2, 'max_tokens': 100}, {'seed': 1, 'max_tokens': 200}]:
        with pytest.raises(SystemExit, match='Incompatible'):
            cached_cell(tmp_path, f, changed, problems, 0, 1)
    with pytest.raises(SystemExit, match='layout'):
        cached_cell(tmp_path, f, config, problems, 1, 1)
    write([dict(record, experiment=None)])
    with pytest.raises(SystemExit, match='unverified'):
        cached_cell(tmp_path, f, config, problems, 0, 1)


def test_changed_local_weights_change_identity(tmp_path):
    f = tmp_path / 'weights.safetensors'
    f.write_bytes(b'first')
    before = model_identity(str(tmp_path))
    f.write_bytes(b'other')
    assert model_identity(str(tmp_path)) != before


@pytest.mark.parametrize('fail', [False, True])
def test_grid_preserves_visible_devices_and_propagates_failure(tmp_path, fail):
    fake = tmp_path / 'python'
    calls = tmp_path / 'calls'
    fake.write_text('#!/bin/sh\necho "$3:$CUDA_VISIBLE_DEVICES" >> "$CALLS"\n'
                    'if [ "$3" = eval ]; then exit "$FAIL"; fi\n')
    fake.chmod(0o755)
    data = tmp_path / 'probe.jsonl'
    data.write_text('{}\n{}\n')
    env = dict(os.environ, PYTHON=str(fake), CALLS=str(calls), FAIL=str(int(fail)),
               CUDA_VISIBLE_DEVICES='4,5', GPUS='2', TP='1', MODEL='olmo7b', BENCH=str(data),
               BATCHES='1', SHARD='1', OUT=str(tmp_path / 'out'))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', str(root / 'scripts/eval_grid.sh')], env=env, capture_output=True)
    lines = calls.read_text().splitlines()
    assert {'eval:4', 'eval:5'}.issubset(lines)
    assert (result.returncode != 0) == fail
    assert any(s.startswith('summarize:') for s in lines) == (not fail)


def test_sandbox_fails_closed(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('docker')
    monkeypatch.setattr(exec_grade.subprocess, 'run', missing)
    with pytest.raises(SystemExit, match='No code was executed'):
        exec_grade.require_sandbox()


def test_sandbox_command_has_no_host_directory_mount():
    cmd = exec_grade.sandbox_command('/tmp/example.py', 'test-container', 'python3')
    for arg in ['--network=none', '--read-only', '--cap-drop=ALL', '--user=65534:65534',
                '--security-opt=no-new-privileges', '--pull=never']:
        assert arg in cmd
    assert cmd[cmd.index('--mount') + 1] == 'type=bind,src=/tmp/example.py,dst=/program.py,readonly'
    assert cmd.count('--mount') == 1


def test_sandbox_timeout_removes_container(monkeypatch):
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == 'run':
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(exec_grade.subprocess, 'run', run)
    assert exec_grade.run_program('pass', timeout=1) == (False, 'timeout')
    assert calls[-1][:3] == ['docker', 'rm', '--force']
