"""Opt-in child-process recovery; leaves checkpointed training files intact."""
import os
import sys

if os.environ.get('FIRST_RUN_HOST_CACHE_RECOVERY') == '1':
    try:
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        os.environ.setdefault('FIRST_RUN_MONITOR_OWNER_PID', str(os.getpid()))
        from host_memory_recovery import (
            emit,
            install_cache_cleanup,
            install_checkpoint_cleanup,
            install_torchrun_logs,
            start_monitor,
        )

        install_torchrun_logs(os.environ.get(
            'FIRST_RUN_RECOVERY_LOG_DIR', '/kaggle/working/first-run-recovery'))

        torch = None
        if 'LOCAL_RANK' in os.environ:
            import torch

            from workflow_distillation.training_checkpoints import CheckpointManager

            install_cache_cleanup(torch)
            install_checkpoint_cleanup(torch, CheckpointManager)
            emit({'event': 'host_cache_recovery_installed', 'rank': os.environ['LOCAL_RANK'],
                  'torch_version': torch.__version__})
        start_monitor(torch)
    except Exception as exc:
        # Python otherwise swallows sitecustomize errors and starts training unpatched.
        print('Recovery startup failed: ' + type(exc).__name__, file=sys.stderr, flush=True)
        os._exit(78)
