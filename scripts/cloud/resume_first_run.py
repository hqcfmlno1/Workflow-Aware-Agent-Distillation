"""Resume checkpoint375 with a fixed, strictly below28k untrained suffix.

Run from the private recovery-code Kaggle dataset, in the existing GPU notebook.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

ARCHIVE_SHA256 = 'c4f05b86a83808db2ca03a25acb2428f44d65e04ab704729e327d25a1a0be9b2'
INDEX_SHA256 = 'd0103eb4eaa2aed12e0d85cc9b952b54ee6a0c9d6536c365fd2a67296a7ccd0e'
SCRIPT_SHA256 = 'e5d71b7cb9cca971d1ce5d2195b1a9da94b1d20f319f1d8fb9f64959940cc22a'
PLAN_SHA256 = '42d0a70cec3f3688da0064524cc5a6e18919db1ac8f16299c37109648db261a3'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_continuation(assets, project, index, checkpoint):
    """Validate the fixed schedule and resume boundary before model allocation."""
    from filtered_continuation import check_resume_manifest, load_plan, validate_plan

    plan = load_plan(assets / 'continuation-plan.json', expected_sha256=PLAN_SHA256)
    validate_plan(index, plan)
    kind = check_resume_manifest(checkpoint, plan)
    entrypoint = project / 'scripts/training/train_qlora_filtered.py'
    shutil.copyfile(assets / 'train_qlora_filtered.py', entrypoint)
    selection = {k: v for k, v in plan.items() if k not in ('ordered_indices', 'parent')}
    selection['parent_progress'] = plan['parent']['progress']
    selection['resume_kind'] = kind
    print(json.dumps({'event': 'continuation_plan_verified', **selection}), flush=True)
    return entrypoint, selection


def run():
    # Set before importing huggingface_hub in this process or child processes.
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    assets = Path(__file__).resolve().parent
    archive = assets / 'original-source.bin'
    if sha256(archive) != ARCHIVE_SHA256:
        raise RuntimeError('Original training archive checksum differs')
    project = Path('/kaggle/working/first_run_recovery_project')
    if project.exists():
        raise RuntimeError('Recovery directory already exists. Do not start duplicate training.')
    # Secret first, before dependency install or model load; never print its value.
    from kaggle_secrets import UserSecretsClient

    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
    print(json.dumps({'run_name': 'first run', 'status': 'secret_loaded',
                      'recovery': 'filtered_suffix_below28000'}), flush=True)
    project.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as stream:
        stream.extractall(project, filter='data')
    entrypoint = project / 'scripts/training/train_qlora_epoch.py'
    if sha256(entrypoint) != SCRIPT_SHA256:
        raise RuntimeError('Original training source differs; refusing checkpoint migration')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r',
                           str(project / 'cloud/kaggle/weighted-requirements.txt')])
    sys.path.insert(0, str(project / 'src'))
    from workflow_distillation.training_checkpoints import download_latest_checkpoint

    base_args = json.loads((assets / 'training-arguments.json').read_text())
    index = Path(base_args[base_args.index('--index') + 1])
    if not index.is_file() or sha256(index) != INDEX_SHA256:
        raise RuntimeError('Original epoch dataset is missing or index differs')
    print(json.dumps({'event': 'checkpoint_download_started', 'repo_type': 'dataset',
                      'repo': 'hqcfmlno1/QwenStacy-4b-coding'}), flush=True)
    checkpoint = download_latest_checkpoint(
        'hqcfmlno1/QwenStacy-4b-coding', project / 'hub-recovery',
        prefix='first-run', repo_type='dataset')
    entrypoint, selection = prepare_continuation(assets, project, index, checkpoint)
    saved = json.loads((checkpoint / 'manifest.json').read_text())
    print(json.dumps({'event': 'recovery_checkpoint_verified',
                      'progress': saved['progress'], 'checkpoint': checkpoint.name}), flush=True)
    log_dir = Path('/kaggle/working/first-run-recovery')
    log_dir.mkdir(exist_ok=True)
    record = {'original_archive_sha256': ARCHIVE_SHA256,
              'original_script_sha256': SCRIPT_SHA256,
              'operational_hooks': {name: sha256(assets / name) for name in
                                    ['sitecustomize.py', 'host_memory_recovery.py',
                                     'recovery_diagnostics.py', 'adaptive_memory.py']},
              'continuation_assets': {name: sha256(assets / name) for name in
                                      ['filtered_continuation.py', 'train_qlora_filtered.py',
                                       'continuation-plan.json']},
              'data_selection': selection,
              'numerical_sources_modified': False, 'checkpoint_progress': saved['progress'],
              'execution_policy': {
                  'backend': 'manual_data_parallel', 'ddp_reducer_created': False,
                  'ddp_bucket_bytes': 0,
                  'modes': ['parallel', 'serial'], 'host_reserve_gib': 3,
                  'activation_estimate_margin': 1.15,
                  'gradient_reduction_both_modes': 'scaled SUM/world_size before unscale',
                  'gradient_reduction_order_may_differ': True,
                  'global_batch_size': 2, 'optimizer_updates_per_batch': 1,
                  'gpu_memory_safety_guaranteed': False,
              },
              'change': 'empty inactive pinned cache before forward and checkpoint; '
                        'full file telemetry, one-minute progress summaries, '
                        'abort owned torchrun groups on new host OOM kills; '
                        'abort on rank failed events; choose parallel/serial before '
                        'retaining host activations; broadcast adapters and average '
                        'gradients without constructing a DDP reducer; preserve consumed '
                        'prefix at375 and filter only pending full trajectories below28000',
              'cause_status': 'recovery9 completed371-376 and published375; '
                              'serial rank0 backward at377 CUDA OOM on29991 tokens; '
                              'host oom_kill counter stayed zero'}
    (log_dir / 'recovery-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    previous_pythonpath = os.environ.get('PYTHONPATH')
    extra_pythonpath = os.pathsep + previous_pythonpath if previous_pythonpath else ''
    env = {**os.environ, 'FIRST_RUN_HOST_CACHE_RECOVERY': '1',
           'FIRST_RUN_ADAPTIVE_MEMORY': '1',
           'FIRST_RUN_CONTINUATION_PLAN': str(assets / 'continuation-plan.json'),
           'FIRST_RUN_CONTINUATION_PLAN_SHA256': PLAN_SHA256,
           'PYTHONFAULTHANDLER': '1',
           'FIRST_RUN_RECOVERY_LOG_DIR': str(log_dir),
           'PYTHONPATH': str(assets) + os.pathsep + str(project / 'src') +
                        extra_pythonpath}
    # Exact parent migration; subsequent checkpoints bind the fixed plan and new entrypoint.
    code = subprocess.call([sys.executable, '-u', str(entrypoint), *base_args,
                            '--resume-from', str(checkpoint)], cwd=project, env=env)
    print(json.dumps({'event': 'recovery_process_finished', 'exit_code': code}), flush=True)
    raise SystemExit(code)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        run()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            raise
        # Capture after the launcher has failed: do not race checkpoint publication.
        from recovery_diagnostics import collect_diagnostics, publish_diagnostics, redact

        secret = os.environ.get('HF_TOKEN', '')
        log_dir = Path('/kaggle/working/first-run-recovery')
        try:
            report = collect_diagnostics(
                '/kaggle/working/first_run_recovery_project', log_dir,
                exit_code=exc.code if isinstance(exc, SystemExit) else 1)
            report['bootstrap_traceback'] = traceback.format_exc()
            payload = redact(json.dumps(report, ensure_ascii=False, indent=2), secret)
            log_dir.mkdir(exist_ok=True)
            (log_dir / 'failure-diagnostics.json').write_text(payload + '\n')
            print(json.dumps({'event': 'recovery_failure_diagnostics',
                              'report': json.loads(payload)}), flush=True)
            if secret:
                publish_diagnostics(json.loads(payload), 'hqcfmlno1/QwenStacy-4b-coding',
                                    secret=secret)
                print(json.dumps({'event': 'recovery_diagnostics_published',
                                  'path': 'README.md', 'checkpoint_paths_modified': False}),
                      flush=True)
        except Exception as diagnostic_error:
            print(json.dumps({'event': 'recovery_diagnostics_failed',
                              'error_type': type(diagnostic_error).__name__}), flush=True)
        raise


if __name__ == '__main__':
    main()
