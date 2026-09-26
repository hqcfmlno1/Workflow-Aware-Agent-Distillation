"""Build OmniCode images with SWE-ReX preinstalled and verify the runtime binary."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_SWE_REX_VERSION = "1.3.0"
DEFAULT_TAG_SUFFIX = "-swerex"
DEFAULT_DOCKERFILE = Path("docker/Dockerfile.swerex")


def _derived_image_tag(base_image: str, tag_suffix: str = DEFAULT_TAG_SUFFIX) -> str:
    if not tag_suffix:
        raise ValueError("tag_suffix must not be empty")
    return base_image if base_image.endswith(tag_suffix) else f"{base_image}{tag_suffix}"


def _docker_build_command(
    base_image: str,
    derived_image: str,
    swe_rex_version: str,
    dockerfile: str,
    context: str = ".",
) -> list[str]:
    return [
        "docker",
        "build",
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"SWE_REX_VERSION={swe_rex_version}",
        "-t",
        derived_image,
        "-f",
        dockerfile,
        context,
    ]


def _docker_preflight_command(image: str) -> list[str]:
    return ["docker", "run", "--rm", image, "swerex-remote", "--version"]


def _docker_retag_command(source_image: str, target_image: str) -> list[str]:
    return ["docker", "tag", source_image, target_image]


def prebuild_image(
    base_image: str,
    *,
    swe_rex_version: str = DEFAULT_SWE_REX_VERSION,
    tag_suffix: str = DEFAULT_TAG_SUFFIX,
    dockerfile: Path = DEFAULT_DOCKERFILE,
    context: Path = Path("."),
    dry_run: bool = False,
    retag_base: bool = True,
) -> str:
    derived_image = _derived_image_tag(base_image, tag_suffix)
    build_command = _docker_build_command(
        base_image,
        derived_image,
        swe_rex_version,
        str(dockerfile),
        str(context),
    )
    preflight_command = _docker_preflight_command(derived_image)
    retag_command = _docker_retag_command(derived_image, base_image)
    if dry_run:
        print(" ".join(build_command))
        print(" ".join(preflight_command))
        if retag_base:
            print(" ".join(retag_command))
        return derived_image

    subprocess.run(build_command, check=True)
    subprocess.run(preflight_command, check=True)
    if retag_base:
        subprocess.run(retag_command, check=True)
    return derived_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Existing OmniCode image names to derive from.")
    parser.add_argument("--tag-suffix", default=DEFAULT_TAG_SUFFIX)
    parser.add_argument("--swe-rex-version", default=DEFAULT_SWE_REX_VERSION)
    parser.add_argument("--dockerfile", type=Path, default=DEFAULT_DOCKERFILE)
    parser.add_argument("--context", type=Path, default=Path("."))
    parser.add_argument(
        "--no-retag",
        action="store_true",
        help="Keep the -swerex tag only; by default also retag the original OmniCode name.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for image in args.images:
        derived = prebuild_image(
            image,
            swe_rex_version=args.swe_rex_version,
            tag_suffix=args.tag_suffix,
            dockerfile=args.dockerfile,
            context=args.context,
            dry_run=args.dry_run,
            retag_base=not args.no_retag,
        )
        print(f"prebuilt_image={derived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
