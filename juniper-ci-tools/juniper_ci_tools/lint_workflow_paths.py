"""Lint workflow-script-path references in ``.github/workflows/*.yml``.

This module is the library implementation behind the
``juniper-lint-workflow-paths`` console script. It is the consolidated
home for what was previously copy-pasted as
``util/test_workflow_script_paths.py`` (or ``tests/...``) into every
Juniper ecosystem repo -- 6 byte-identical copies before this
consolidation.

What it catches
---------------

Every ``python|bash <path/to/script>`` invocation in a
``.github/workflows/*.yml`` file must reference a path that exists in
the repo. Catches the failure class that broke 3 juniper-X CIs on
2026-05-18, where a script was renamed (or its symlink target moved
into a non-checked-out sibling repo) but the workflow continued to
invoke the old path. The CI would fail with
``python: can't open file '.../scripts/check_doc_links.py'`` on every
run until somebody noticed.

What it does NOT catch
----------------------

- Module form (``python -m foo.bar``) -- cannot resolve a module to a
  path without importing the package.
- Cross-repo paths (``juniper-X/...``) -- these are runtime-resolved
  from sibling clones in scheduled workflows.
- Absolute paths (``/usr/local/bin/foo``).
- Shell-variable-expanded paths (``${{ env.SCRIPT }}/foo.py``).

Working directories
-------------------

A ``run:`` step executes in its effective working directory, so a path
must be resolved against that directory and not only against the repo
root. Precedence, highest first (GitHub's own):

1. ``jobs.<id>.steps[].working-directory``
2. ``jobs.<id>.defaults.run.working-directory``
3. ``defaults.run.working-directory`` (workflow level)
4. the repo root

A path is reported missing only when it exists at **neither** its
effective working directory **nor** the repo root. That is deliberately
permissive: paths also appear in strings that are not ``run:`` bodies
(``with:`` inputs, ``if:`` expressions), where no working directory
applies, and this lint exists to catch renames -- a rename removes the
file from both locations, so nothing it was built for slips through.

Before 2026-09-09 the resolution was ``repo_root / path`` unconditionally.
That made every monorepo lane with a nested package a false positive:
juniper-recurrence's ``ci-recurrence-model.yml`` sets
``working-directory: juniper-recurrence-model`` and runs
``pytest tests/test_readouts_mlp.py``, which exists -- and the lane was
green while the lint disagreed (juniper-ml#1836). The hazard was the
suggested repair: prefixing the path in the workflow *breaks* the lane,
because pytest would then look for
``juniper-recurrence-model/juniper-recurrence-model/tests/...``.

Library API
-----------

::

    from juniper_ci_tools.lint_workflow_paths import (
        lint_workflow_paths,
        LintFinding,
        LintResult,
        ScriptReference,
        find_repo_root,
        extract_script_paths,
        extract_script_references,
    )

    result = lint_workflow_paths(repo_root)  # auto-discovers via .github/workflows
    if not result.ok:
        for finding in result.missing:
            print(f"{finding.workflow}: {finding.path}")

Console script
--------------

::

    juniper-lint-workflow-paths [--repo-root PATH] [--workflows-dir PATH]
                                [--exit-zero] [--json] [--version]

Exit codes: ``0`` (no missing paths), ``1`` (missing paths found, default;
suppress with ``--exit-zero``), ``2`` (repo root or workflows dir not
discoverable).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import yaml

# Path-like script references: a token containing at least one ``/`` and
# ending in .py/.sh/.bash, not preceded by another path char. Catches all
# the common invocation forms:
#
#   python scripts/check_doc_links.py
#   python3 -m unittest -v tests/test_foo.py
#   bash util/generate_dep_docs.sh
#   $PYTHON scripts/foo.py            (shell var prefix)
#
# Module form ``python -m foo.bar`` is intentionally not validated because
# we cannot resolve a module to a path without importing the package.
#
# Note on the inner character class: the per-segment class
# ``[A-Za-z0-9_.-]+`` deliberately **excludes** the ``/`` character so each
# outer ``(?:/...)`` iteration consumes exactly one path segment. The
# earlier form that included ``/`` in the inner class was a CodeQL
# `py/redos` finding (overlap between the outer ``+`` group and the inner
# ``+`` quantifier permitted exponential backtracking on adversarial
# inputs starting with ``-/`` repeated).
_SCRIPT_PATH = re.compile(r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|sh|bash))\b")

# Sibling ecosystem repos that scheduled workflows clone into the runner
# workspace before invoking. Any path under one of these is runtime-
# resolved from a clone, not a path checked into this repo, so the lint
# must skip it.
DEFAULT_ECOSYSTEM_SIBLING_PREFIXES: tuple[str, ...] = (
    "juniper-canopy/",
    "juniper-cascor/",
    "juniper-cascor-client/",
    "juniper-cascor-worker/",
    "juniper-data/",
    "juniper-data-client/",
    "juniper-deploy/",
    "juniper-ml/",
)


@dataclass(frozen=True)
class ScriptReference:
    """A script path as referenced, together with the directory it runs in.

    ``working_directory`` is the effective ``working-directory`` for the step the
    path was found in, relative to the repo root -- ``""`` when none applies
    (the repo root itself, or a string outside any step).
    """

    path: str
    working_directory: str = ""

    def candidates(self) -> tuple[str, ...]:
        """Repo-root-relative locations this reference may legitimately resolve to."""
        if not self.working_directory:
            return (self.path,)
        return (f"{self.working_directory}/{self.path}", self.path)


@dataclass(frozen=True)
class LintFinding:
    """A single workflow-script-path lint finding."""

    workflow: Path
    """The workflow file that references the missing path."""

    path: str
    """The (relative) path that does not exist in the repo."""

    working_directory: str = ""
    """The effective ``working-directory`` the path was resolved against, if any.

    Reported so the reader can see *where* the lint looked. Without it the
    obvious repair for a monorepo lane is to prefix the path in the workflow,
    which breaks the lane (juniper-ml#1836).
    """


@dataclass(frozen=True)
class LintResult:
    """Aggregate lint result over all workflow files."""

    repo_root: Path
    workflows_dir: Path
    workflow_files: tuple[Path, ...]
    missing: tuple[LintFinding, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return len(self.missing) == 0

    def report(self) -> str:
        """Human-readable summary suitable for CLI output or
        ``unittest.TestCase.fail`` argument."""
        if self.ok:
            return f"OK: {len(self.workflow_files)} workflow file(s) checked under {self.workflows_dir}, no missing script paths."
        lines = [
            "CI workflow(s) reference script paths that do not exist:",
            *(f"  {f.workflow.relative_to(self.repo_root)}: references missing path '{f.path}'" + (f" (searched '{f.working_directory}/{f.path}' and '{f.path}')" if f.working_directory else "") for f in self.missing),
            "",
            "This is the failure class that broke 3 juniper-X CIs on 2026-05-18 (script rename without workflow update).",
            "Either restore the missing path or update the workflow.",
            "Where a working directory is shown, both locations were checked: do NOT 'fix' this by",
            "prefixing the path in the workflow -- the step already runs in that directory, so the",
            "prefix would be applied twice and break the lane.",
        ]
        return "\n".join(lines)


def _iter_yaml_strings(node: object) -> Iterator[str]:
    """Yield every string value reachable in a parsed YAML tree."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _iter_yaml_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_yaml_strings(value)


def extract_script_paths(yaml_text: str) -> set[str]:
    """Extract ``python <path.py>`` and ``bash <path.{bash,sh}>`` paths
    from a workflow YAML file's parsed content. Returns the set of
    raw path strings (before any validatability filtering).

    Workflows that fail to parse are silently treated as having no
    extractable paths -- the YAML error is its own concern and would
    surface from yamllint / actionlint / GitHub.
    """
    paths: set[str] = set()
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return paths

    for value in _iter_yaml_strings(parsed):
        for match in _SCRIPT_PATH.finditer(value):
            paths.add(match.group(1))
    return paths


def _paths_in(node: object) -> set[str]:
    """Every script path reachable in a YAML subtree."""
    found: set[str] = set()
    for value in _iter_yaml_strings(node):
        for match in _SCRIPT_PATH.finditer(value):
            found.add(match.group(1))
    return found


def _working_directory(container: object) -> Optional[str]:
    """``defaults.run.working-directory`` of a workflow- or job-level mapping."""
    if not isinstance(container, dict):
        return None
    run = (container.get("defaults") or {}).get("run") if isinstance(container.get("defaults"), dict) else None
    if isinstance(run, dict):
        wd = run.get("working-directory")
        if isinstance(wd, str) and wd.strip():
            return wd.strip().rstrip("/")
    return None


def extract_script_references(yaml_text: str) -> set[ScriptReference]:
    """Extract script paths from a workflow, each tagged with the working
    directory the step that references it actually runs in.

    Walks the ``jobs`` -> ``steps`` structure rather than flattening the tree, so
    ``working-directory`` context survives. Strings outside any step (top-level
    ``env``, ``on``, a job's ``container``, ...) are still collected, with no
    working directory -- dropping them would lose coverage the flat extractor had.

    A workflow that fails to parse yields nothing; the YAML error is its own
    concern and surfaces from yamllint / actionlint / GitHub.
    """
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return set()
    if not isinstance(parsed, dict):
        return set()

    workflow_wd = _working_directory(parsed)
    refs: set[ScriptReference] = set()

    jobs = parsed.get("jobs")
    if isinstance(jobs, dict):
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            job_wd = _working_directory(job) or workflow_wd
            steps = job.get("steps")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                step_wd = step.get("working-directory")
                effective = step_wd.strip().rstrip("/") if isinstance(step_wd, str) and step_wd.strip() else (job_wd or "")
                for path in _paths_in(step):
                    refs.add(ScriptReference(path=path, working_directory=effective))

    # Anything not inside a step keeps the flat extractor's behaviour: repo-root
    # relative, no working directory. Computed as a set difference so a path that
    # appears both in a step and elsewhere keeps its working-directory-tagged form.
    in_steps = {ref.path for ref in refs}
    for path in extract_script_paths(yaml_text) - in_steps:
        refs.add(ScriptReference(path=path, working_directory=workflow_wd or ""))

    return refs


def is_validatable(
    path: str,
    *,
    sibling_prefixes: tuple[str, ...] = DEFAULT_ECOSYSTEM_SIBLING_PREFIXES,
) -> bool:
    """Filter out paths the lint cannot resolve to an on-disk file."""
    if "${" in path or "$(" in path:  # shell-expanded variables
        return False
    if path.startswith("/"):  # absolute paths (e.g., toolcache python)
        return False
    if path.startswith("-"):  # caught a flag like ``-m``
        return False
    if path.startswith(sibling_prefixes):
        return False  # cross-repo path resolved at runtime, not at lint time
    # Skip standalone short filenames (likely a shell variable or a
    # runtime-extracted name) -- we only validate paths that include a
    # directory.
    return "/" in path


def find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` looking for the first ancestor that
    contains a ``.github/workflows/`` directory. That's the repo root
    relative to which every workflow path resolves.

    Raises :class:`RuntimeError` if no such ancestor exists.
    """
    for parent in [start, *start.parents]:
        if (parent / ".github" / "workflows").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repo root: no .github/workflows/ directory found in any ancestor of {start}")


def lint_workflow_paths(
    repo_root: Path,
    *,
    workflows_dir: Optional[Path] = None,
    sibling_prefixes: tuple[str, ...] = DEFAULT_ECOSYSTEM_SIBLING_PREFIXES,
) -> LintResult:
    """Walk every ``*.yml`` / ``*.yaml`` file under
    ``repo_root/.github/workflows/`` (or ``workflows_dir`` if given) and
    return a :class:`LintResult` listing every script path that is
    referenced but does not exist on disk.

    ``sibling_prefixes`` lets consumers tune the cross-repo skip list
    (e.g., for repos outside the Juniper ecosystem). Defaults match the
    8-repo Juniper layout that motivated this module.
    """
    wf_dir = workflows_dir if workflows_dir is not None else repo_root / ".github" / "workflows"
    workflow_files = tuple(sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")))

    missing: list[LintFinding] = []
    for wf_file in workflow_files:
        text = wf_file.read_text(encoding="utf-8")
        for ref in extract_script_references(text):
            if not is_validatable(ref.path, sibling_prefixes=sibling_prefixes):
                continue
            # Missing only when it resolves NOWHERE -- neither under the step's
            # effective working directory nor at the repo root. See the module
            # docstring on why that permissiveness is the right trade.
            if not any((repo_root / candidate).exists() for candidate in ref.candidates()):
                missing.append(LintFinding(workflow=wf_file, path=ref.path, working_directory=ref.working_directory))

    return LintResult(
        repo_root=repo_root,
        workflows_dir=wf_dir,
        workflow_files=workflow_files,
        missing=tuple(missing),
    )


__all__ = [
    "DEFAULT_ECOSYSTEM_SIBLING_PREFIXES",
    "LintFinding",
    "LintResult",
    "ScriptReference",
    "extract_script_paths",
    "extract_script_references",
    "find_repo_root",
    "is_validatable",
    "lint_workflow_paths",
]
