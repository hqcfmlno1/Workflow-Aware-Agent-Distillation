"""Preserve bounded, redacted failure evidence in the private dataset card."""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

START = '<!-- first-run-diagnostics:start -->'
END = '<!-- first-run-diagnostics:end -->'


def redact(text, secret=''):
    if secret:
        text = text.replace(secret, '[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9]{10,}', '[REDACTED]', text)


def collect_diagnostics(project, log_dir, *, exit_code, cgroup_root=Path('/sys/fs/cgroup')):
    from host_memory_recovery import cgroup_memory

    project, log_dir = Path(project), Path(log_dir)
    output = project / 'outputs/training/first run'
    paths = [output / name for name in ['first run.log', 'report.json',
             'rank0-summary.json', 'rank1-summary.json', 'rank0.jsonl', 'rank1.jsonl']]
    paths += sorted(log_dir.rglob('*.log')) + sorted(log_dir.rglob('error.json'))
    paths += sorted(log_dir.glob('memory-*.jsonl'))
    report = {'time': datetime.now(UTC).isoformat(), 'exit_code': exit_code,
              'cgroup_after': cgroup_memory(cgroup_root), 'files': {}}
    remaining = 100_000
    for path in paths:
        if not path.is_file() or remaining <= 0:
            continue
        count = min(remaining, 12_000)
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - count))
            tail = stream.read(count)
        report['files'][str(path)] = tail.decode(errors='replace')
        remaining -= len(tail)
    return report


def diagnostic_readme(existing, report, *, secret=''):
    pattern = re.escape(START) + r'.*?' + re.escape(END)
    original = re.sub(pattern, '', existing, flags=re.DOTALL).rstrip()
    payload = redact(json.dumps(report, ensure_ascii=False, indent=2), secret)
    # JSON strings can contain Markdown fences; keep the report in a longer fence.
    fence = '`' * max(4, max((len(m[0]) for m in re.finditer(r'`+', payload)), default=0) + 1)
    return (original + '\n\n' + START + '\n<details>\n'
            '<summary>Latest first run failure diagnostics</summary>\n\n' +
            fence + 'json\n' + payload + '\n' + fence + '\n\n</details>\n' + END + '\n')


def publish_diagnostics(report, repo_id, *, secret=''):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.repo_info(repo_id, repo_type='dataset')
    if not info.private:
        raise ValueError('Failure diagnostics require a private dataset')
    existing = ''
    if 'README.md' in api.list_repo_files(repo_id, repo_type='dataset', revision=info.sha):
        path = hf_hub_download(repo_id, 'README.md', repo_type='dataset', revision=info.sha)
        existing = Path(path).read_text()
    content = diagnostic_readme(existing, report, secret=secret)
    api.upload_file(repo_id=repo_id, repo_type='dataset', path_in_repo='README.md',
                    path_or_fileobj=content.encode(), parent_commit=info.sha,
                    commit_message='Record latest first run failure diagnostics')
