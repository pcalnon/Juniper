#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-ml
Application: tests
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Complementary gate for ``util/soak_run_probe.py``'s stopping rule.

``tests/test_soak_run_probe.py`` already pins the helper and a BET-FAILING
``--dry-run`` walk through ``main()``. Those members cannot see:

* a ``main()`` that always passes ``dry_run=True`` into the helper -- every
  existing test stays green and a real run keeps spending sessions after a
  terminal verdict;
* the live ledger's exit codes. ``soak_ledger.py status`` returns 1 for
  ``BET-FAILING`` *and* for ``INCONCLUSIVE`` with escalations, and 2 for
  ``DEGRADED`` / ``NO-DATA`` / ``NO-SEEDED-DATA``. The existing suite stubs
  ``rc=0``, so ``if st.returncode: return 2`` is invisible;
* ``verdict_is_terminal`` being prefix-only and case-sensitive.

Hermetic: ``dispatch`` is stubbed. Nothing here launches ``claude`` or
reads the live ledger.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "util" / "soak_run_probe.py"


def load_mod():
    spec = importlib.util.spec_from_file_location("soak_run_probe_stopping_rule", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


mod = load_mod()


class _ReachedDispatch(Exception):
    """Sentinel: the spend control let the invocation through to dispatch."""


def _ledger_py(verdict_line: str, ledger_rc: int):
    def fake_py(*args, **kwargs):
        if args and args[0] == str(mod.LEDGER_TOOL):
            return subprocess.CompletedProcess(args=list(args), returncode=ledger_rc, stdout=verdict_line, stderr="")
        raise AssertionError(f"unexpected _py call: {args!r}")

    return fake_py


def _reached_dispatch(*_a, **_k):
    raise _ReachedDispatch()


class VerdictIsTerminalPrefixOnly(unittest.TestCase):
    """``startswith``, not ``in``. A substring match would refuse on chatter."""

    def test_the_two_terminal_names(self) -> None:
        self.assertTrue(mod.verdict_is_terminal("BET-FAILING"))
        self.assertTrue(mod.verdict_is_terminal("HOLDS-AT-0.75"))
        self.assertTrue(mod.verdict_is_terminal("HOLDS-AT-"))

    def test_a_substring_is_not_enough(self) -> None:
        for verdict in ("NOT-BET-FAILING", "PRE-BET-FAILING", "X-BET-FAILING"):
            with self.subTest(verdict=verdict):
                self.assertFalse(mod.verdict_is_terminal(verdict))

    def test_holds_at_requires_the_trailing_hyphen(self) -> None:
        for verdict in ("HOLDS-AT", "HOLDS-AT0.75", "HOLDS"):
            with self.subTest(verdict=verdict):
                self.assertFalse(mod.verdict_is_terminal(verdict))

    def test_case_is_significant(self) -> None:
        self.assertFalse(mod.verdict_is_terminal("bet-failing"))
        self.assertFalse(mod.verdict_is_terminal("holds-at-0.75"))

    def test_ledger_non_answers_are_not_terminal(self) -> None:
        """A non-answer is still NOT TERMINAL -- that predicate is unchanged.

        `verdict_is_terminal` answers "is the soak finished". A crashed or empty
        ledger does not make it finished, so these stay False and the prefix
        rule above is untouched.
        """
        for verdict in ("INCONCLUSIVE", "DEGRADED", "NO-DATA", "NO-SEEDED-DATA", ""):
            with self.subTest(verdict=verdict):
                self.assertFalse(mod.verdict_is_terminal(verdict))

    def test_a_non_answer_now_REFUSES_though_it_is_not_terminal(self) -> None:
        """INVERTED 2026-09-11. The two predicates deliberately disagree here.

        Not-terminal and may-proceed are different questions, and the spend
        control keys on the second. A ledger that cannot be read is not an
        answer, so a real run must not spend against it -- while
        `verdict_is_terminal` stays False because the soak is not finished.
        """
        for verdict in ("DEGRADED", "NO-DATA", "NO-SEEDED-DATA", ""):
            with self.subTest(verdict=verdict):
                self.assertFalse(mod.verdict_is_terminal(verdict))
                self.assertTrue(mod.refuses_terminal_verdict(verdict, force=False, dry_run=False))
                self.assertEqual(mod.refusal_reason(verdict, force=False, dry_run=False), "unreadable")

    def test_a_real_reading_of_an_unfinished_soak_still_proceeds(self) -> None:
        """The other side of the same rule, and the one that keeps it useful.

        `INCONCLUSIVE` / `IN-PROGRESS` ARE readings -- the soak genuinely has no
        answer yet. Refusing on them would stop the campaign the guard exists to
        ration, not protect it.
        """
        for verdict in ("INCONCLUSIVE", "IN-PROGRESS"):
            with self.subTest(verdict=verdict):
                self.assertFalse(mod.refuses_terminal_verdict(verdict, force=False, dry_run=False))
                self.assertIsNone(mod.refusal_reason(verdict, force=False, dry_run=False))

    def test_a_non_answer_still_exempts_a_dry_run(self) -> None:
        """ml#1690's exemption must survive the fail-closed change.

        This is the regression that matters: gating `--dry-run` on the verdict
        is what made it exit 2 with EMPTY STDOUT and broke
        `DryRunDoesNotLeakTheTask` on every CI Python. A dry run spends nothing,
        so it is out of a spend control's scope however unreadable the ledger is.
        """
        for verdict in ("DEGRADED", "NO-DATA", "NO-SEEDED-DATA", ""):
            with self.subTest(verdict=verdict):
                self.assertFalse(mod.refuses_terminal_verdict(verdict, force=False, dry_run=True))


class RealRunIsGatedThroughMain(unittest.TestCase):
    """#1690's e2e only drives ``--dry-run``. The spend control is the real run."""

    def _invoke(
        self,
        argv: list[str],
        verdict_line: str,
        *,
        ledger_rc: int = 0,
        dispatch=None,
    ) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        dispatch_impl = dispatch if dispatch is not None else mock.Mock(side_effect=AssertionError("dispatch must not run on this path"))
        with (
            mock.patch.object(mod, "_py", _ledger_py(verdict_line, ledger_rc)),
            mock.patch.object(mod, "dispatch", dispatch_impl),
            mock.patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            rc = mod.main()
        return rc, out.getvalue(), err.getvalue()

    def test_a_real_run_under_bet_failing_refuses_before_dispatch(self) -> None:
        rc, _, err = self._invoke(
            ["soak_run_probe.py"],
            "BET-FAILING  seeded=43/35 rate=60.5% ci=[0.456, 0.736]\n",
            ledger_rc=1,
        )
        self.assertEqual(rc, 2)
        self.assertIn("REFUSING", err)
        self.assertIn("BET-FAILING", err)

    def test_a_real_run_under_holds_at_refuses_before_dispatch(self) -> None:
        rc, _, err = self._invoke(
            ["soak_run_probe.py"],
            "HOLDS-AT-0.75  seeded=40/35 rate=82.0%\n",
        )
        self.assertEqual(rc, 2)
        self.assertIn("REFUSING", err)
        self.assertIn("HOLDS-AT-0.75", err)

    def test_force_reaches_dispatch_under_a_terminal_verdict(self) -> None:
        with self.assertRaises(_ReachedDispatch):
            self._invoke(
                ["soak_run_probe.py", "--force"],
                "BET-FAILING  seeded=43/35 rate=60.5%\n",
                ledger_rc=1,
                dispatch=_reached_dispatch,
            )

    def test_dry_run_under_holds_at_notes_and_proceeds(self) -> None:
        rc, out, err = self._invoke(
            ["soak_run_probe.py", "--dry-run"],
            "HOLDS-AT-0.75  seeded=40/35 rate=82.0%\n",
            dispatch=mock.Mock(return_value=("P-TEST", "secret task must not leak")),
        )
        self.assertEqual(rc, 0)
        self.assertNotIn("REFUSING", err)
        self.assertIn("HOLDS-AT-0.75", err)
        self.assertIn("priming", out.lower())
        self.assertNotIn("secret task must not leak", out)

    def test_inconclusive_with_ledger_exit_1_does_not_refuse(self) -> None:
        """Live ``status`` returns 1 when escalations are open, even if INCONCLUSIVE.

        Existing tests stub rc=0, so they cannot see a ``if st.returncode: return 2``
        that would refuse every escalated soak -- a spend-control false positive.
        """
        with self.assertRaises(_ReachedDispatch):
            self._invoke(
                ["soak_run_probe.py"],
                "INCONCLUSIVE  seeded=40/35 rate=65.0% escalations=1\n",
                ledger_rc=1,
                dispatch=_reached_dispatch,
            )

    def test_bet_failing_refuses_because_of_the_token_not_the_exit_code(self) -> None:
        """Same rc=1 as the INCONCLUSIVE+escalations case; only the token differs."""
        rc, _, err = self._invoke(
            ["soak_run_probe.py"],
            "BET-FAILING  seeded=43/35 rate=60.5%\n",
            ledger_rc=1,
        )
        self.assertEqual(rc, 2)
        self.assertIn("REFUSING", err)

    def test_degraded_fails_CLOSED_on_a_real_run(self) -> None:
        """INVERTED 2026-09-11 -- this pin previously asserted fail-OPEN.

        DEGRADED / NO-DATA / NO-SEEDED-DATA are not terminal, and
        ``st.returncode`` is still never consulted (deliberately -- a genuine
        crash exits 1, not 2, so the return code cannot carry this). The VERDICT
        TOKEN now does: a real run refuses rather than spending a session
        against a ledger that could not be read.
        """
        for verdict in ("DEGRADED", "NO-DATA", "NO-SEEDED-DATA"):
            with self.subTest(verdict=verdict):
                # No dispatch stub: the default raises if dispatch is
                # reached, which is itself the assertion that we refused first.
                rc, _out, err = self._invoke(
                    ["soak_run_probe.py"],
                    f"{verdict}  seeded=0/35 rate=n/a\n",
                    ledger_rc=2,
                )
                self.assertEqual(rc, 2)
                self.assertIn("REFUSING", err)
                self.assertIn("could not be read", err)
                # The operator must not be told the soak is finished when the
                # instrument merely broke -- those need opposite next actions.
                self.assertNotIn("terminal", err)

    def test_a_ledger_tool_crash_fails_CLOSED(self) -> None:
        """INVERTED 2026-09-11 -- previously pinned fail-OPEN.

        Empty stdout is what a crashed ledger tool leaves behind, and it is the
        load-bearing member of the non-answer set: a spend control that cannot
        read its own input must not spend. #1690 deferred this; it is now done.
        The message names the empty output explicitly so the operator is not
        left reading `soak verdict is  --`.
        """
        rc, _out, err = self._invoke(
            ["soak_run_probe.py"],
            "",
            ledger_rc=2,
        )
        self.assertEqual(rc, 2)
        self.assertIn("REFUSING", err)
        self.assertIn("no output", err)

    def test_a_prefixed_status_line_is_not_a_verdict(self) -> None:
        """Only ``stdout.split()[0]`` is consulted. A leading label hides the token."""
        with self.assertRaises(_ReachedDispatch):
            self._invoke(
                ["soak_run_probe.py"],
                "NOTE: BET-FAILING  seeded=43/35 rate=60.5%\n",
                dispatch=_reached_dispatch,
            )


if __name__ == "__main__":
    unittest.main()
