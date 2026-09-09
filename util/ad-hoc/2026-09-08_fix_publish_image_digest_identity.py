#!/usr/bin/env python3
"""
Replace the digest-identity block of a publish-image.yml merge job with one that resolves each
pushed per-arch digest to its image manifest(s) before comparing it to the published manifest list.

Project: juniper-ml
Sub-Project: ad-hoc tooling
Author: Paul Calnon
Created: 2026-09-08
Status: ad-hoc — one-off (five byte-identical copies of the block, one correction)
Retire when: the corrected block is on main in all five image-bearing repos (worker, data, cascor,
             canopy, recurrence) and their publish paths have each run green once.
Related: juniper-cascor-worker dispatch run 34292278641 (the failure this corrects);
         juniper-ml prompts/thread-handoff_automated-prompts/
         HANDOFF_2026-09-07_container-registry-rollout-wave-1-complete.md item 1.

Why: the merge job's "Verify published image is CPU-only" step compared the digests the build
jobs pushed (``steps.build.outputs.digest``) with the ``.manifests[].digest`` entries of the
published tag. With provenance attestations on -- docker/build-push-action's default -- each
pushed digest names a per-arch OCI INDEX (image manifest + attestation), and
``imagetools create`` flattens those indexes into the tag, so the tag's entries are the IMAGE
manifests. The two sets can never be equal; the step failed on the very first publish that ran
it, AFTER the tag had been written. Verified against the live registry: the pushed digests
95e605af… (arm64) and ac848506… (amd64) each contain exactly one linux image manifest, and those
two are exactly the tag's linux entries (ee9064e2…, ae1d5377…).

Usage::

    python3 util/ad-hoc/2026-09-08_fix_publish_image_digest_identity.py PATH [PATH ...]

Exit 0 when every file had exactly one old block and was rewritten; 1 otherwise (no file is
touched unless ALL of them match, so a partial rollout cannot happen by accident).
"""

from __future__ import annotations

import sys
from pathlib import Path

OLD_BLOCK = """          echo "--- digest identity"
          listed="$(jq -r '[.manifests[]? | select(.platform.os=="linux") | .digest] | sort | join(" ")' /tmp/manifest.json)"
          verified="$(cat /tmp/digests/* | LC_ALL=C sort | tr '\\n' ' ' | sed 's/ $//')"
          if [ "${listed}" != "${verified}" ]; then
            echo "::error::manifest list references [${listed}] but the verified build digests are [${verified}]"
            exit 1
          fi
          echo "  ${ref} references exactly the verified digests"
"""

NEW_BLOCK = """          echo "--- digest identity"
          # Each build job pushed its arch BY DIGEST, and with provenance attestations on (the
          # build-push-action default) that digest names a per-arch OCI INDEX -- image manifest plus
          # attestation -- which `imagetools create` FLATTENS into the published list. So the list's
          # entries are the IMAGE manifests, not the pushed index digests, and comparing the two
          # directly fails on every publish (worker dispatch run 34292278641, 2026-09-08). Resolve
          # each verified digest to the image manifest(s) it carries first; a digest that already IS
          # an image manifest (provenance off) has no `.manifests` and stands for itself.
          listed="$(jq -r '[.manifests[]? | select(.platform.os=="linux") | .digest] | sort | join(" ")' /tmp/manifest.json)"
          verified=""
          for f in /tmp/digests/*; do
            pushed="$(cat "${f}")"
            images="$(docker buildx imagetools inspect "${REGISTRY}/${IMAGE_NAME}@${pushed}" --format '{{ json .Manifest }}' \\
              | jq -r '[.manifests[]? | select(.platform.os=="linux") | .digest] | join(" ")')"
            verified="${verified} ${images:-${pushed}}"
          done
          verified="$(tr ' ' '\\n' <<< "${verified}" | sed '/^$/d' | LC_ALL=C sort | tr '\\n' ' ' | sed 's/ $//')"
          if [ "${listed}" != "${verified}" ]; then
            echo "::error::manifest list references [${listed}] but the verified build digests resolve to [${verified}]"
            exit 1
          fi
          echo "  ${ref} references exactly the verified images"
"""


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1
    texts: dict[Path, str] = {}
    ok = True
    for raw in paths:
        path = Path(raw)
        text = path.read_text(encoding="utf-8")
        count = text.count(OLD_BLOCK)
        if count != 1:
            print(f"!! {path}: expected exactly one old digest-identity block, found {count}")
            ok = False
        texts[path] = text
    if not ok:
        print("no file touched")
        return 1
    for path, text in texts.items():
        path.write_text(text.replace(OLD_BLOCK, NEW_BLOCK), encoding="utf-8")
        print(f"rewrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
