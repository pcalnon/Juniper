#!/usr/bin/env python3
"""
Harden the two in-repo CPU-only checks after the refuter lane of the 2026-09-08 handoff review:
(1) the in-image census also forbids ``cuda-*`` distributions, (2) the merge job's digest-identity
step asserts exactly one linux image per pushed per-arch digest, whose architecture matches the
digest file's name.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-08
Status: ad-hoc — one-off (five byte-identical copies of each block, one correction)
Retire when: both hardenings are on main in all five image-bearing repos.
Related: juniper-ml prompts/thread-handoff_automated-prompts/
         HANDOFF_2026-09-08_container-registry-rollout-wave-2-opened-and-the-cuda-class-in-three-shapes.md
         (Validation record, Lane B2 findings 2 and 3).

Why (1): the real CUDA image (dispatch-3d81f2c) carried ``cuda-toolkit``, ``cuda-bindings`` and
``cuda-pathfinder`` alongside the ``nvidia-*`` wheels and ``triton``. The census forbade only the
latter two families, so an image with torch==X+cpu plus those three would have read
``cuda_stack=0 … CPU-only contract holds`` -- while the handoff's own verification grep counts
``^cuda-`` as CUDA stack. Exposure today is ~6.6 MB (no current lock pins them), but the check must
mean what the contract says.

Why (2): the identity step resolves each pushed digest to the linux image(s) it carries and compares
the SET with the tag's linux entries. A pushed index carrying two linux images (a build job given two
platforms) would count both as "verified" although the census ran on the runner's platform only.
``/tmp/digests/<arch>`` makes the missing assertion free: one linux image, ``architecture`` == file name.

Usage::

    python3 2026-09-08_harden_publish_image_checks.py --workflow PATH ... --census PATH ... [--tests PATH ...]

Every named file must contain exactly one old block, or nothing is written (no partial rollout).
``--tests`` extends the census-behaviour test and pins the per-digest assertion in the three
torch-bearing repos' ``test_dockerfile_cpu_torch_pin.py`` (data and recurrence have no such file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OLD_LOOP = """          for f in /tmp/digests/*; do
            pushed="$(cat "${f}")"
            images="$(docker buildx imagetools inspect "${REGISTRY}/${IMAGE_NAME}@${pushed}" --format '{{ json .Manifest }}' \\
              | jq -r '[.manifests[]? | select(.platform.os=="linux") | .digest] | join(" ")')"
            verified="${verified} ${images:-${pushed}}"
          done
"""

NEW_LOOP = """          for f in /tmp/digests/*; do
            pushed="$(cat "${f}")"
            arch_file="$(basename "${f}")"
            manifest_json="$(docker buildx imagetools inspect "${REGISTRY}/${IMAGE_NAME}@${pushed}" --format '{{ json .Manifest }}')"
            images="$(jq -r '[.manifests[]? | select(.platform.os=="linux") | .digest] | join(" ")' <<< "${manifest_json}")"
            if [ -n "${images}" ]; then
              # A pushed index must carry exactly ONE linux image, and it must be the arch the build
              # job (and its census) ran on -- the digest file is named after that arch. Otherwise a
              # multi-platform index would count an image the census never saw as "verified".
              image_arch="$(jq -r '[.manifests[]? | select(.platform.os=="linux") | .platform.architecture] | join(" ")' <<< "${manifest_json}")"
              if [ "$(wc -w <<< "${images}")" -ne 1 ] || [ "${image_arch}" != "${arch_file}" ]; then
                echo "::error::pushed digest ${pushed} (digest file '${arch_file}') carries linux image(s) for [${image_arch}] -- expected exactly one, for ${arch_file}"
                exit 1
              fi
            fi
            verified="${verified} ${images:-${pushed}}"
          done
"""

OLD_CENSUS = """_CPU_VERSION_RE = re.compile(r"^\\d+\\.\\d+\\.\\d+\\+cpu$")
_FORBIDDEN_PREFIX = "nvidia-"
_FORBIDDEN_NAMES = frozenset({"triton"})
"""

NEW_CENSUS = """_CPU_VERSION_RE = re.compile(r"^\\d+\\.\\d+\\.\\d+\\+cpu$")
# The whole CUDA stack as it actually appeared in the 2026-09-07 image: the nvidia-* runtime
# libraries, triton, AND the cuda-* packages (cuda-toolkit, cuda-bindings, cuda-pathfinder).
# The first census forbade only the first two families and would have passed the third.
_FORBIDDEN_PREFIXES = ("nvidia-", "cuda-")
_FORBIDDEN_NAMES = frozenset({"triton"})
"""

OLD_FORBIDDEN_FN = """    return sorted(n for n in names if n.startswith(_FORBIDDEN_PREFIX) or n in _FORBIDDEN_NAMES)
"""

NEW_FORBIDDEN_FN = """    return sorted(n for n in names if n.startswith(_FORBIDDEN_PREFIXES) or n in _FORBIDDEN_NAMES)
"""

OLD_DOC = """``<major>.<minor>.<patch>+cpu``
    torch must import, ``torch.__version__`` must equal this value exactly,
    ``torch.version.cuda`` must be ``None``, and no ``nvidia-*`` or ``triton``
    distribution may be installed.

``absent``
    torch must not be importable, and no ``nvidia-*`` / ``triton`` distribution may
    be installed (for images that never ship torch).
"""

NEW_DOC = """``<major>.<minor>.<patch>+cpu``
    torch must import, ``torch.__version__`` must equal this value exactly,
    ``torch.version.cuda`` must be ``None``, and no ``nvidia-*``, ``cuda-*`` or
    ``triton`` distribution may be installed.

``absent``
    torch must not be importable, and no ``nvidia-*`` / ``cuda-*`` / ``triton``
    distribution may be installed (for images that never ship torch).
"""

# The census-behaviour test in the three torch-bearing repos' test_dockerfile_cpu_torch_pin.py:
# extend the sample with two cuda-* names (one underscore-spelled, to go through _normalise) and
# add one explicit regression test for the family the first census let through.
OLD_TEST_CENSUS = """        names = {"numpy", "nvidia-cublas", "nvidia_cudnn_cu13", "triton", "sympy"}
        offenders = mod.forbidden_distributions({mod._normalise(n) for n in names})
        assert offenders == ["nvidia-cublas", "nvidia-cudnn-cu13", "triton"]
"""

NEW_TEST_CENSUS = '''        names = {"numpy", "nvidia-cublas", "nvidia_cudnn_cu13", "triton", "cuda-toolkit", "cuda_bindings", "sympy"}
        offenders = mod.forbidden_distributions({mod._normalise(n) for n in names})
        assert offenders == ["cuda-bindings", "cuda-toolkit", "nvidia-cublas", "nvidia-cudnn-cu13", "triton"]

    def test_census_forbids_the_cuda_prefix_family(self):
        """The 2026-09-07 CUDA image also carried cuda-toolkit / cuda-bindings / cuda-pathfinder; the first census let those through."""
        mod = _load_check_module()
        assert mod.forbidden_distributions({"cuda-toolkit", "cuda-bindings", "cuda-pathfinder", "numpy"}) == ["cuda-bindings", "cuda-pathfinder", "cuda-toolkit"]
'''

# The merge-job test: pin the new per-digest assertion right after the existing identity assertion.
OLD_TEST_IDENTITY = """        assert "/tmp/manifest.json" in step["run"] and "/tmp/digests" in step["run"], "digest identity between the manifest list and the verified digests must be asserted"
"""

NEW_TEST_IDENTITY = '''        assert "/tmp/manifest.json" in step["run"] and "/tmp/digests" in step["run"], "digest identity between the manifest list and the verified digests must be asserted"

    def test_identity_step_admits_one_linux_image_per_pushed_digest(self):
        """A pushed per-arch digest names an OCI index; it must carry exactly one linux image, for the arch the census ran on (the digest file's name)."""
        run = _step("merge", "Verify published image is CPU-only")["run"]
        assert 'arch_file="$(basename "${f}")"' in run, "the expected arch is the digest file's name"
        assert '"${image_arch}" != "${arch_file}"' in run, "the linux image's architecture must equal the digest file's name"
        assert "-ne 1" in run, "exactly one linux image per pushed digest"
'''


def _apply(paths: list[Path], pairs: list[tuple[str, str]], label: str) -> bool:
    ok = True
    texts: dict[Path, str] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for old, _ in pairs:
            count = text.count(old)
            if count != 1:
                print(f"!! {path}: {label}: expected exactly one occurrence of a block, found {count}")
                ok = False
        texts[path] = text
    if not ok:
        return False
    for path, text in texts.items():
        for old, new in pairs:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        print(f"rewrote {path} ({label})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workflow", nargs="*", default=[], help="publish-image.yml paths")
    ap.add_argument("--census", nargs="*", default=[], help="check_image_cpu_only.py paths")
    ap.add_argument("--tests", nargs="*", default=[], help="test_dockerfile_cpu_torch_pin.py paths (torch-bearing repos only)")
    args = ap.parse_args()
    if not args.workflow and not args.census and not args.tests:
        print(__doc__)
        return 1
    ok_w = _apply([Path(p) for p in args.workflow], [(OLD_LOOP, NEW_LOOP)], "identity loop") if args.workflow else True
    ok_c = _apply([Path(p) for p in args.census], [(OLD_CENSUS, NEW_CENSUS), (OLD_FORBIDDEN_FN, NEW_FORBIDDEN_FN), (OLD_DOC, NEW_DOC)], "census") if args.census else True
    ok_t = _apply([Path(p) for p in args.tests], [(OLD_TEST_CENSUS, NEW_TEST_CENSUS), (OLD_TEST_IDENTITY, NEW_TEST_IDENTITY)], "tests") if args.tests else True
    return 0 if (ok_w and ok_c and ok_t) else 1


if __name__ == "__main__":
    sys.exit(main())
