import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cues import demo


def test_dry_run_dispatches_without_gpu_dependencies(tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'cues', 'demo', '--model', 'olmo7b', '--dry-run'],
        env=dict(os.environ, RUNS=str(tmp_path / 'runs with spaces')),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line) for line in result.stdout.splitlines() if ' -m cues ' in line]
    assert [cmd[3] for cmd in commands] == ['eval', 'summarize', 'table']
    evaluation = commands[0]
    assert evaluation[evaluation.index('--out') + 1] == str(tmp_path / 'runs with spaces' / 'demo')
    assert not (tmp_path / 'runs with spaces').exists()


def test_demo_orchestrates_existing_commands(monkeypatch, tmp_path, capsys):
    calls = []
    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(demo.subprocess, 'run', run)
    output = tmp_path / 'custom output'
    demo.main(['--model', 'qwen14b', '--tensor-parallel-size', '2', '--out', str(output)])
    evaluation, summary, table = calls
    assert evaluation[evaluation.index('--tensor-parallel-size') + 1] == '2'
    assert evaluation[evaluation.index('--cues') + 1:evaluation.index('--cues') + 3] == ['none', 'auto']
    assert evaluation[evaluation.index('--problem-count') + 1] == '5'
    assert evaluation[evaluation.index('--n-rollouts') + 1] == '1'
    assert evaluation[evaluation.index('--max-new-tokens') + 1] == '2048'
    assert summary[-2:] == ['--label', 'qwen14b']
    assert '--rollouts' in summary
    assert table[4] == str(output)
    report = capsys.readouterr().out
    assert "' Alright,'" in report
    assert 'not a full benchmark' in report
    assert shlex.join(table) in report


@pytest.mark.parametrize('failed_stage', [0, 1, 2])
def test_failed_stage_stops_pipeline_without_touching_outputs(monkeypatch, tmp_path, failed_stage):
    sentinel = tmp_path / 'existing.jsonl'
    sentinel.write_text('keep me')
    calls = []
    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=7 if len(calls) - 1 == failed_stage else 0)
    monkeypatch.setattr(demo.subprocess, 'run', run)
    with pytest.raises(SystemExit) as error:
        demo.main(['--model', 'olmo7b', '--out', str(tmp_path)])
    assert error.value.code == 7
    assert len(calls) == failed_stage + 1
    assert sentinel.read_text() == 'keep me'


@pytest.mark.parametrize('extra', [['--tensor-parallel-size', '0'], ['--tensor-parallel-size', '-2'], ['--unknown']])
def test_invalid_arguments_do_not_start_evaluation(monkeypatch, extra):
    def unexpected(command):
        pytest.fail('Invalid input started evaluation')
    monkeypatch.setattr(demo.subprocess, 'run', unexpected)
    with pytest.raises(SystemExit) as error:
        demo.main(['--model', 'olmo7b', *extra])
    assert error.value.code == 2
