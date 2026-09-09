#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-ml (cascor#573 logging redesign, ROADMAP decision 5 / P4.1)
Application: ad-hoc measurement
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Purpose: measure the per-call cost of the FOUR candidate dispatch mechanisms for
cascor's Logger, so P4.1's "class-based vs instance-based vs both" choice is taken
against numbers instead of intuition.

Why it matters: the emit methods are called 646,016 times per 32-profile cap-4
corpus (RECON 6.1) and ~91% of those calls are discarded by the level filter. Any
per-call dispatch overhead is therefore paid ~646k times per corpus, on the path the
whole arc has been making cheaper.

Mechanisms measured:
  A  @classmethod                      -- the status quo
  B  plain instance method             -- pure per-instance design
  C  hybrid descriptor, via the CLASS   -- Logger.debug(...)
  D  hybrid descriptor, via an INSTANCE -- self.logger.debug(...)
  E  module-level function             -- floor, for scale

Each body is trivial and identical (a level compare + early return), so the delta
IS the dispatch cost. Reported as ns/call and as seconds per 646,016-call corpus.

Read-only; touches no repo state.
"""
import sys
import timeit

N = 400_000
CORPUS_CALLS = 646_016


class _Hybrid:
    """Descriptor that binds to the instance when there is one, else to the class.

    This is the mechanism a dual-access ("both class and object calls") Logger needs:
    ``__get__`` receives obj=None for class access and the instance otherwise, so a
    single implementation can serve both. The cost is that a NEW bound object is
    built on every attribute access -- which is what this benchmark prices.
    """

    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        target = objtype if obj is None else obj
        func = self.func

        def bound(*args, **kwargs):
            return func(target, *args, **kwargs)

        return bound


class _HybridPartial:
    """Same contract, but binds with functools.partial instead of a closure."""

    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        from functools import partial

        return partial(self.func, objtype if obj is None else obj)


THRESHOLD = 20


class ClassStyle:
    _level = THRESHOLD

    @classmethod
    def debug(cls, level, message):
        if level < cls._level:
            return None
        return message


class InstanceStyle:
    def __init__(self):
        self._level = THRESHOLD

    def debug(self, level, message):
        if level < self._level:
            return None
        return message


def _hybrid_body(target, level, message):
    if level < target._level:
        return None
    return message


class HybridStyle:
    _level = THRESHOLD

    debug = _Hybrid(_hybrid_body)
    debug_partial = _HybridPartial(_hybrid_body)

    def __init__(self):
        self._level = THRESHOLD


def module_debug(level, message):
    if level < THRESHOLD:
        return None
    return message


def bench(label, stmt, setup_globals):
    t = timeit.timeit(stmt, globals=setup_globals, number=N)
    ns = t / N * 1e9
    corpus = ns * CORPUS_CALLS / 1e9
    return label, ns, corpus


def main():
    inst = InstanceStyle()
    hyb = HybridStyle()
    g = {
        "ClassStyle": ClassStyle,
        "InstanceStyle": InstanceStyle,
        "HybridStyle": HybridStyle,
        "inst": inst,
        "hyb": hyb,
        "module_debug": module_debug,
    }

    # level 1 (TRACE) against a threshold of 20 -> the DISCARDED path, i.e. 91% of calls
    rows = [
        bench("E  module-level function            ", 'module_debug(1, "m")', g),
        bench("A  @classmethod        (status quo) ", 'ClassStyle.debug(1, "m")', g),
        bench("B  plain instance method            ", 'inst.debug(1, "m")', g),
        bench("C  hybrid descriptor, via CLASS     ", 'HybridStyle.debug(1, "m")', g),
        bench("D  hybrid descriptor, via INSTANCE  ", 'hyb.debug(1, "m")', g),
        bench("D' hybrid via partial, via INSTANCE ", 'hyb.debug_partial(1, "m")', g),
    ]

    base = [r for r in rows if r[0].startswith("A ")][0][1]
    print(f"n={N:,} iterations per row; discarded-record path (level 1 vs threshold {THRESHOLD})")
    print(f"corpus column = cost over {CORPUS_CALLS:,} logger calls (RECON 6.1, one 32-profile cap-4 corpus)\n")
    print(f"{'mechanism':<38} {'ns/call':>9} {'vs @classmethod':>17} {'s / corpus':>12}")
    print("-" * 80)
    for label, ns, corpus in rows:
        delta = f"{ns - base:+.1f} ns" if not label.startswith("A ") else "--"
        print(f"{label:<38} {ns:>9.1f} {delta:>17} {corpus:>12.4f}")
    print("-" * 80)

    print("\nSanity: all mechanisms agree on the discarded path and the emitted path.")
    checks = [
        ClassStyle.debug(1, "m"), inst.debug(1, "m"), HybridStyle.debug(1, "m"),
        hyb.debug(1, "m"), hyb.debug_partial(1, "m"), module_debug(1, "m"),
    ]
    emitted = [
        ClassStyle.debug(30, "m"), inst.debug(30, "m"), HybridStyle.debug(30, "m"),
        hyb.debug(30, "m"), hyb.debug_partial(30, "m"), module_debug(30, "m"),
    ]
    print(f"  discarded -> {checks}  (all None: {all(c is None for c in checks)})")
    print(f"  emitted   -> {emitted}  (all 'm': {all(e == 'm' for e in emitted)})")

    print("\nIdentity check -- does a hybrid access allocate a NEW object each time?")
    a, b = HybridStyle.debug, HybridStyle.debug
    print(f"  HybridStyle.debug is HybridStyle.debug  -> {a is b}")
    c, d = ClassStyle.debug, ClassStyle.debug
    print(f"  ClassStyle.debug  is ClassStyle.debug   -> {c is d}   (bound classmethods are also rebuilt)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
