"""A failed recovery must retain useful diagnostics without exposing credentials."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HOOKS = Path(__file__).resolve().parents[1] / 'cloud/kaggle/recovery'


def load_diagnostics(monkeypatch):
    monkeypatch.syspath_prepend(str(HOOKS))
    spec = importlib.util.spec_from_file_location('recovery_diagnostics',
                                                 HOOKS / 'recovery_diagnostics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collects_rank_error_and_post_failure_oom(monkeypatch, tmp_path):
    module = load_diagnostics(monkeypatch)
    project, logs, cgroup = (tmp_path / name for name in ['project', 'logs', 'cgroup'])
    for folder in [project, logs / 'torchrun/job/attempt_0/1', cgroup]:
        folder.mkdir(parents=True)
    (logs / 'torchrun/job/attempt_0/1/stderr.log').write_text('CUDA load failed\n')
    (cgroup / 'memory.events').write_text('oom 2\noom_kill 1\n')
    report = module.collect_diagnostics(project, logs, exit_code=1, cgroup_root=cgroup)
    assert report['exit_code'] == 1
    assert report['cgroup_after']['events']['oom_kill'] == 1
    assert any('CUDA load failed' in value for value in report['files'].values())


def test_readme_preserves_card_replaces_previous_report_and_redacts(monkeypatch):
    module = load_diagnostics(monkeypatch)
    original = '---\nlicense: other\n---\nUser description.\n'
    secret = 'hf_' + 'x' * 32
    first = module.diagnostic_readme(original, {'error': secret}, secret=secret)
    assert secret not in first
    assert first.startswith(original.rstrip())
    second = module.diagnostic_readme(first, {'error': 'new report'}, secret=secret)
    assert second.count(module.START) == 1
    assert 'new report' in second
    assert 'User description.' in second


def test_upload_uses_private_repo_and_revision_without_changing_checkpoint_paths(
        monkeypatch, tmp_path):
    module = load_diagnostics(monkeypatch)
    calls = []
    api = SimpleNamespace(
        repo_info=lambda *args, **kwargs: SimpleNamespace(private=True, sha='original-sha'),
        list_repo_files=lambda *args, **kwargs: ['README.md', 'first-run/latest.json'],
        upload_file=lambda **kwargs: calls.append(kwargs))
    card = tmp_path / 'README.md'
    card.write_text('Original dataset card.\n')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: api, hf_hub_download=lambda *args, **kwargs: str(card)))
    module.publish_diagnostics({'exit_code': 1}, 'owner/private', secret='test-secret')
    assert len(calls) == 1
    assert calls[0]['path_in_repo'] == 'README.md'
    assert calls[0]['parent_commit'] == 'original-sha'
    assert 'Original dataset card.' in calls[0]['path_or_fileobj'].decode()
    api.repo_info = lambda *args, **kwargs: SimpleNamespace(private=False, sha='public-sha')
    with pytest.raises(ValueError, match='private'):
        module.publish_diagnostics({'exit_code': 1}, 'owner/public', secret='test-secret')
    assert len(calls) == 1
