#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-ml (cascor#573 logging redesign, ROADMAP decision 5 / P4.1)
Application: ad-hoc measurement
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Purpose: price option **A2** -- "two entry points, ONE implementation" -- against the
status quo, so decision 5 can be taken on a number rather than on my estimate.

A2 keeps ``Logger``'s classmethods but makes them thin DELEGATORS to a default
per-logger instance, so there is literally one logging code path reachable both as
``Logger.debug(...)`` and as ``self.logger.debug(...)``. Option A instead keeps two
separate implementations. The only thing separating them is what the delegation
costs on the CLASS path.

Why this script exists separately from 2026-09-09_p41_dualpath_mechanisms_bench.py:
that one passes the level as an argument to isolate raw dispatch. This one uses
juniper-cascor's REAL public signature --

    @classmethod
    def debug(cls, message=None, *args) -> None          # logger.py:548-555

-- because the delegator has to FORWARD ``*args``, and the earlier benchmark already
showed ``*args``/``**kwargs`` packing dominating closure dispatch (99 ns fixed-signature
vs 243-288 ns variadic). An estimate that ignores forwarding would be wrong in the
direction that matters.

Rows:
  SQ-class   status quo, class access          -- what ALL 1,188 sites cost today,
                                                  since `self.logger = Logger` binds
                                                  the CLASS (candidate_unit.py:187)
  A2-class   A2 delegator, class access        -- what the 109 direct-class sites become
  A2-lazy    A2 delegator with a lazy-init guard (bootstrap safety, constraint 3)
  A/A2-inst  plain instance method             -- what the 967 self.logger sites become
  A-class    option A's untouched classmethod  -- identical to SQ-class by construction

Reported as the range across REPEATS_OUTER independent runs; the host is shared.
Read-only; touches no repo state.
"""
import sys
import timeit

N = 200_000
REPEATS = 7
REPEATS_OUTER = 5
CORPUS_CALLS = 646_016

DEBUG = 10          # the level the method implies
THRESHOLD = 20      # configured INFO -> a debug() call is DISCARDED, as 91% are


# --------------------------------------------------------------- the status quo
class StatusQuo:
    """Today: one class, class-level state, 32 classmethods."""

    _level = THRESHOLD

    @classmethod
    def debug(cls, message=None, *args) -> None:
        if DEBUG < cls._level:
            return None
        return message


# ------------------------------------------------- the shared A/A2 instance path
class BoundLogger:
    """The per-logger instance both A and A2 hand to call sites."""

    __slots__ = ("_name", "_level")

    def __init__(self, name, level=THRESHOLD):
        self._name = name
        self._level = level

    def debug(self, message=None, *args) -> None:
        if DEBUG < self._level:
            return None
        return message


# ------------------------------------------------------------------- option A2
class A2Logger:
    """Classmethods are thin delegators to a default instance: ONE implementation."""

    _default = BoundLogger("root")

    @classmethod
    def debug(cls, message=None, *args) -> None:
        return cls._default.debug(message, *args)


class A2LazyLogger:
    """As A2, but the default is created on first use.

    Constraint 3 (bootstrap): logger.py logs during its own construction, before any
    instance can exist. A module-level default may not be constructible that early,
    so the realistic delegator may need this guard. It costs one None test per call.
    """

    _default = None

    @classmethod
    def debug(cls, message=None, *args) -> None:
        d = cls._default
        if d is None:
            d = cls._default = BoundLogger("root")
        return d.debug(message, *args)


class A2BindLogger:
    """A2 without the delegating CALL: the class attribute IS the default's bound method.

    ``Logger.debug`` is not a classmethod here -- it is a plain class attribute holding
    ``_default.debug`` already bound. A class-path call is then an attribute lookup plus
    the same instance-method call the hot path makes, with no second frame and no
    ``*args`` repacking. Still ONE implementation.

    The costs are not performance: rebinding the default (``set_level`` on the root,
    reconfiguration) must REASSIGN these attributes or they keep pointing at the old
    instance; the attributes are not introspectable as methods; and a subclass inherits
    a binding to the PARENT's default. Whether those are acceptable is a design
    question, not a measurement -- this row only settles whether the speed is there.
    """

    _default = BoundLogger("root")
    debug = _default.debug


def bench(stmt, g):
    """min over REPEATS; the distribution is one-sided so the minimum estimates cost."""
    return min(timeit.repeat(stmt, globals=g, number=N, repeat=REPEATS)) / N * 1e9


def main():
    bound = BoundLogger("candidate_unit")
    A2LazyLogger.debug("warm")  # pay the lazy init once, so the row is steady-state
    g = {
        "StatusQuo": StatusQuo, "A2Logger": A2Logger,
        "A2LazyLogger": A2LazyLogger, "bound": bound,
        "A2BindLogger": A2BindLogger,
    }

    rows = [
        ("SQ-class   status quo (all 1,188 sites today)", 'StatusQuo.debug("m")'),
        ("A/A2-inst  plain instance method (967 sites)  ", 'bound.debug("m")'),
        ("A2-bind    bound method as class attr (109)   ", 'A2BindLogger.debug("m")'),
        ("A2-class   delegator, eager default (109)     ", 'A2Logger.debug("m")'),
        ("A2-lazy    delegator, lazy default (109)      ", 'A2LazyLogger.debug("m")'),
    ]
    # one extra arg, to expose *args forwarding cost in the delegator
    rows_args = [
        ("SQ-class   + 1 arg                            ", 'StatusQuo.debug("m", 1)'),
        ("A/A2-inst  + 1 arg                            ", 'bound.debug("m", 1)'),
        ("A2-bind    + 1 arg                            ", 'A2BindLogger.debug("m", 1)'),
        ("A2-class   + 1 arg                            ", 'A2Logger.debug("m", 1)'),
    ]

    print(f"n={N:,} x {REPEATS} repeats, best-of; x {REPEATS_OUTER} outer runs -> range")
    print(f"DISCARDED path (debug={DEBUG} vs threshold {THRESHOLD}) -- 91% of real calls")
    print(f"python {sys.version.split()[0]}\n")

    def measure(rowset):
        acc = {label: [] for label, _ in rowset}
        for _ in range(REPEATS_OUTER):
            for label, stmt in rowset:
                acc[label].append(bench(stmt, g))
        return acc

    def report(acc, base_label):
        base_lo = min(acc[base_label])
        print(f"{'row':<46} {'ns/call':>13} {'vs status quo':>16} {'s / corpus':>12}")
        print("-" * 92)
        for label in acc:
            lo, hi = min(acc[label]), max(acc[label])
            rng = f"{lo:.0f}-{hi:.0f}"
            delta = "--" if label == base_label else f"{lo - base_lo:+.0f} ns"
            print(f"{label:<46} {rng:>13} {delta:>16} {lo * CORPUS_CALLS / 1e9:>12.4f}")
        print("-" * 92)
        return {k: min(v) for k, v in acc.items()}

    print("=== no extra args (the common case: 775 of the live sites are f-strings) ===")
    got = report(measure(rows), "SQ-class   status quo (all 1,188 sites today)")

    print("\n=== with one %-arg (what P6.4 would convert sites TO) ===")
    report(measure(rows_args), "SQ-class   + 1 arg                            ")

    sq = got["SQ-class   status quo (all 1,188 sites today)"]
    inst = got["A/A2-inst  plain instance method (967 sites)  "]
    a2b = got["A2-bind    bound method as class attr (109)   "]
    a2c = got["A2-class   delegator, eager default (109)     "]
    a2l = got["A2-lazy    delegator, lazy default (109)      "]

    print("\n=== what this means for the real call-site mix ===")
    print("  (all 14 binds move, so the 967 self.logger AND the 112 bare-local sites")
    print("   become instance calls; only the 109 direct-class sites differ by option)")
    print(f"  1,079 sites (bound)       : {sq:6.0f} -> {inst:6.0f} ns  ({inst - sq:+.0f})   [A and A2 alike]")
    print(f"    109 sites (Logger.M)    : {sq:6.0f} -> {sq:6.0f} ns  (   +0)   [A  -- classmethod untouched]")
    print(f"    109 sites (Logger.M)    : {sq:6.0f} -> {a2b:6.0f} ns  ({a2b - sq:+.0f})   [A2-bind]")
    print(f"    109 sites (Logger.M)    : {sq:6.0f} -> {a2c:6.0f} ns  ({a2c - sq:+.0f})   [A2-delegator, eager]")
    print(f"    109 sites (Logger.M)    : {sq:6.0f} -> {a2l:6.0f} ns  ({a2l - sq:+.0f})   [A2-delegator, lazy]")

    print("\n  Where the 109 live -- none is in the per-record loop:")
    print("    64 logger.py (own construction) | 15 main.py | 11 snapshot_utils.py")
    print("     8 spiral_problem.py | 7 log_config.py | 4 cascade_correlation.py")

    print("\nCorrectness -- every path must agree on discard and on emit:")
    loud = BoundLogger("loud", level=DEBUG)
    A2Loud = type("A2Loud", (A2Logger,), {"_default": loud})
    print(f"  discarded: SQ={StatusQuo.debug('m')!r} inst={bound.debug('m')!r} "
          f"A2={A2Logger.debug('m')!r} lazy={A2LazyLogger.debug('m')!r} "
          f"bind={A2BindLogger.debug('m')!r}")
    print(f"  emitted  : inst={loud.debug('m')!r} A2={A2Loud.debug('m')!r}")

    print("\n=== A2-bind: the three properties that decide whether it is safe ===")

    print("\n  1. A MISSED bind degrades gracefully (no TypeError) --")
    print("     a call site left as `self.logger = Logger` still works, on the root level.")

    class _Holder:
        pass

    h = _Holder()
    h.logger = A2BindLogger  # the unconverted bind
    print(f"     h.logger = A2BindLogger; h.logger.debug('m') -> {h.logger.debug('m')!r}  (no error)")

    print("\n  2. Per-instance levels are independent of the class default --")
    noisy = BoundLogger("noisy", level=DEBUG)   # DEBUG enabled
    quiet = BoundLogger("quiet", level=99)      # everything off
    print(f"     noisy.debug('m') -> {noisy.debug('m')!r}   quiet.debug('m') -> {quiet.debug('m')!r}"
          f"   class -> {A2BindLogger.debug('m')!r}")

    print("\n  3. THE HAZARD -- replacing _default does NOT move the bound attribute.")
    before = A2BindLogger.debug("m")
    A2BindLogger._default = loud          # swap in a logger that WOULD emit
    after = A2BindLogger.debug("m")
    print(f"     _default swapped to an emitting logger:  before={before!r}  after={after!r}")
    if after == before:
        print("     -> STALE. The class attribute still points at the ORIGINAL instance.")
        print("        Reconfiguration must MUTATE the default in place, or REASSIGN all")
        print("        eight bound attributes. A `set_level` that rebinds _default is a")
        print("        silent no-op for every class-path call site.")
    else:
        print("     -> followed the swap (unexpected; re-check the mechanism)")
    # restore, so the row above stays reproducible on a re-run
    A2BindLogger._default = BoundLogger("root")
    A2BindLogger.debug = A2BindLogger._default.debug

    print("\n  Mutating in place DOES work, which is the supported reconfiguration path:")
    A2BindLogger._default._level = DEBUG
    print(f"     _default._level = DEBUG -> A2BindLogger.debug('m') = {A2BindLogger.debug('m')!r}")
    A2BindLogger._default._level = THRESHOLD
    return 0


if __name__ == "__main__":
    sys.exit(main())
