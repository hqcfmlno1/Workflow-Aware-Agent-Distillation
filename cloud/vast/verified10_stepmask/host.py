"""Serve only the new masked span-anchor adapter, in an isolated BF16 vLLM engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))


def validate_export(info, protocol):
    source = protocol["sources"]["span-stepmask"]
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision") or ""):
        raise ValueError("Finalize preparation to pin the new checkpoint before serving")
    if info.get("source") != source or info.get("checkpoint") != protocol["checkpoint"]:
        raise ValueError("Wrong adapter source/checkpoint for masked span-anchor")


def verify_bundle():
    inventory = json.loads((HERE / "bundle-manifest.json").read_text())
    for name, checksum in inventory.items():
        with (HERE / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                raise ValueError("Deployment file changed: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    verify_bundle()
    protocol = json.loads((HERE / "protocol.json").read_text())
    from benchmarks.swe_bench_verified.export_adapter import verify_export

    adapter = HERE / "adapters/span-stepmask"
    validate_export(verify_export(adapter), protocol)
    if args.download_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            protocol["base_model"],
            revision=protocol["base_revision"],
            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx", "*.gguf"],
        )
        return
    if not os.environ.get("BENCH_API_KEY"):
        raise ValueError("Set BENCH_API_KEY before GPU loading")
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A GPU supporting native BF16 is required")
    import uvicorn
    from benchmarks.swe_bench_verified.config import BenchConfig, fingerprint, sha256_file
    from benchmarks.swe_bench_verified.serving import create_app
    from benchmarks.swe_bench_verified.vllm_backend import VllmBackend, VllmSettings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = BenchConfig(**protocol["config"])
    backend = VllmBackend(
        config,
        adapter,
        settings=VllmSettings(
            chunk_tokens=config.prefill_chunk_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prefix_caching=not args.no_prefix_caching,
            enforce_eager=args.enforce_eager,
        ),
    )
    try:
        backend.metadata.pop("fingerprint")
        backend.metadata.update(
            benchmark_protocol_sha256=fingerprint(protocol),
            benchmark_host_sha256=sha256_file(__file__),
        )
        backend.metadata["fingerprint"] = fingerprint(backend.metadata)
        (HERE / "server-manifest.json").write_text(json.dumps(backend.metadata, indent=2) + "\n")
        print("MODEL READY: span-stepmask epoch 2 / step 590; BF16; native context", flush=True)
        uvicorn.run(
            create_app(backend, os.environ["BENCH_API_KEY"]),
            host="127.0.0.1",
            port=args.port,
            workers=1,
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
