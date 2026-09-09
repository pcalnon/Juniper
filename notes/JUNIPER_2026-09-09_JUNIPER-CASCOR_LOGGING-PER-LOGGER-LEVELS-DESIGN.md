# Per-logger levels — dispatch design and the dual-access question

- **Project**: Juniper
- **Sub-Project**: juniper-cascor (`src/log_config/`)
- **Author**: Paul Calnon
- **License**: MIT License
- **Version**: 0.7.1
- **Last Updated**: 2026-09-09
- **Status**: DESIGN — analysis and recommendation for P4.1; the implementation design is fleshed out once an approach is selected (§9)
- **Answers**: [`JUNIPER_2026-09-02_JUNIPER-CASCOR_LOGGING-REDESIGN-ROADMAP.md`](JUNIPER_2026-09-02_JUNIPER-CASCOR_LOGGING-REDESIGN-ROADMAP.md) §13 decision 5, and the owner's follow-on question about serving logging as **both** class and instance calls
- **Measured at**: juniper-cascor `origin/main` `53c0338`, Python 3.13.13 (`JuniperCascor1`)

> **Documents referenced here**, by short name, full filename given once:
>
> | short name    | filename, all under `juniper-ml/notes/`                                     |
> |---------------|------------------------------------------------------------------------------|
> | **ROADMAP**   | `JUNIPER_2026-09-02_JUNIPER-CASCOR_LOGGING-REDESIGN-ROADMAP.md`              |
> | **RECON**     | `JUNIPER_2026-09-02_JUNIPER-CASCOR_LOGGING-CURRENT-STATE-RECONCILIATION.md`  |
> | **DESIGN**    | `JUNIPER_2026-08-29_JUNIPER-CASCOR_LOGGING-REDESIGN-DESIGN.md`               |

---

## 1. Why this document exists

ROADMAP §13 decision 5 asks whether "per-logger levels" (#573 scope 3) means **per-instance
loggers** or **named sub-loggers on the class**, and states that per-instance "means
de-classmethod-ing `isEnabledFor` and breaking ~1,200 call sites. This decides whether P4 is M or L."

The owner declined to pick from that framing and asked for the investigation first, plus a specific
further question:

> *"what issues, risks, performance impacts, and concerns would arise from having both class based
> and instance based logging code paths. Having name-keyed level maps, sub-loggers, logging utils,
> etc. accessible as both class and object calls seems like a convenience win. If the fully
> functional logging code path can be utilized both as class based method calls and as per instance
> calls, the cascor modules can optimize their logging based on the specifics of each logging call
> site."*

This document answers both. **Two of the premises the decision rested on turn out to be wrong**, and
correcting them changes the recommendation.

---

## 2. Correction 1 — the break surface is ~45 sites, not ~1,200

ROADMAP §13 decision 5's "~1,200 call sites" counts **call sites**. What actually breaks under an
instance-method design is only the sites that name the **class**, because a site written
`self.logger.debug(...)` is textually identical whether `self.logger` holds the `Logger` class or a
`Logger` instance — only the **bind** changes.

Censused at `53c0338` over tracked `src/`, less `cascade_correlation/backups/` (dead, RECON N-6),
less `src/api/` (stdlib-bound, not Path A, RECON §3.1), less `src/tests/`
(`juniper-ml/util/ad-hoc/2026-09-09_p41_logger_binding_census.py`):

| access syntax          | sites | share  | breaks under instance methods?                            |
|------------------------|------:|-------:|-----------------------------------------------------------|
| `self.logger.M(...)`   | **967** | 81.4 % | **No** — identical either way; only the bind changes      |
| bare `logger.M(...)`   |   112 |  9.4 % | **No** — a local name; survives if the local binds an instance |
| `Logger.M(...)`        | **109** |  9.2 % | **Yes** — direct class access                             |
| `cls.logger.M(...)`    |     0 |    —   | —                                                          |
| **total**              | **1,188** | 100 % | **109 (9.2 %)**                                          |

And of the 109 direct-class sites, **64 are inside `logger.py` itself** — the logger logging about
its own construction, where class access is the natural form and arguably should stay. That leaves
**45 external sites**, in five files:

| file                                      | direct-class sites |
|-------------------------------------------|-------------------:|
| `src/log_config/logger/logger.py`         | 64 *(internal)*    |
| `src/main.py`                             | 15                 |
| `src/snapshots/snapshot_utils.py`         | 11                 |
| `src/spiral_problem/spiral_problem.py`    | 8                  |
| `src/log_config/log_config.py`            | 7                  |
| `src/cascade_correlation/cascade_correlation.py` | 4           |

**Bind sites are 14**, not a sweep: 5 real `self.logger = Logger` (`candidate_unit.py:187`, `:296`,
`cascade_correlation.py:3188`, `cascor_plotter.py:73`, `log_config.py:201` — a sixth grep match is
the docstring at `logger.py:1034`) and 9 `logger = Logger` locals (`cascade_correlation.py:632`,
`:3495`, `:3610`, `:3740`, `:4136`, `:5198`, `cascor_plotter.py:92`, `utils.py:125`, `:273`).

Two further binds reach the class **indirectly** and must not be missed:
`cascade_correlation.py:667` and `spiral_problem.py:310` both do
`self.logger = self.log_config.get_logger()`, and `LogConfig.get_logger` returns `self.logger`,
which `log_config.py:201` set to the `Logger` class.

**Consequence.** Per-instance is an **M**, not an L. The sizing question decision 5 was framed
around does not discriminate between the options.

---

## 3. Correction 2 — dual *syntax* already works; only dual *state* is missing

`@classmethod` is already accessible from an instance. `instance.debug(...)` on a classmethod binds
`cls` to the class and runs fine — measured at **92.0 ns** against **89.7 ns** for `Logger.debug(...)`,
a **+2.3 ns** difference (§4).

So the convenience the owner describes — "accessible as both class and object calls" — is *not* what
is missing today. What is missing is that both access paths reach the **same class-level state**.
Every level knob is a class attribute: `_log_level` (`logger.py:193`), `_level_logger_name` (`:164`),
`_level_logger_config` (`:162`), `_level_number_cache` (`:221`), `_logging_file` (`:140`). `Logger`
carries **32 `@classmethod`s**, and `isEnabledFor`'s own docstring (`:1032-1034`) records the reason:
*"Defined as a classmethod for consistency with all other Logger methods … since Logger is used as a
class-level singleton via `self.logger = Logger`."*

**The real question is therefore not "can both syntaxes work" but "can one NAME dispatch on its
receiver — class state for class access, instance state for instance access — and what does that
cost on a path taken 646,016 times per corpus?"**

---

## 4. What each dispatch mechanism costs

Measured with `timeit.repeat(..., repeat=7)` reporting the **minimum** — the correct estimator here,
because scheduler noise is one-sided and only ever adds time. Bodies are trivial and identical across
mechanisms (a level compare and an early return), so **the delta is the dispatch cost**. Level 1
against a threshold of 20, i.e. the **discarded** path — 91.0 % of real calls (RECON §6.1).

Script: `juniper-ml/util/ad-hoc/2026-09-09_p41_dualpath_mechanisms_bench.py`, with the naive
descriptor in `…/2026-09-09_p41_hybrid_descriptor_bench.py`.

| mechanism                                             | ns/call | vs `@classmethod` | s / 646 k corpus |
|-------------------------------------------------------|--------:|------------------:|-----------------:|
| **B** plain instance method                            |  **48.7** | **−41.0 ns**    | 0.032            |
| **G** metaclass design, via an **instance**            |    87.8 |          −1.9 ns  | 0.057            |
| **A** `@classmethod` — **status quo**                  |    89.7 |               —   | 0.058            |
| **A′** `@classmethod` via an **instance**              |    92.0 |          +2.3 ns  | 0.059            |
| **H** eager-bind in `__init__`, via an instance        |    99.2 |          +9.5 ns  | 0.064            |
| **I** instance-`__dict__` cache, steady state          |   280.2 |        +190.5 ns  | 0.181            |
| **F** metaclass **data descriptor**, via the **class** |   353.4 |        +263.7 ns  | 0.228            |
| **hybrid descriptor** (naive `__get__`), either path   | 441–527 |    +350…+438 ns   | 0.29–0.34        |

**The pattern is unambiguous.** Every mechanism that makes **one name** dispatch on its receiver —
the naive hybrid, the metaclass data descriptor, the instance-`__dict__` cache — costs **3–6× a
classmethod**. Every mechanism that lets the two access paths be **two different bindings** costs
nothing, or is *cheaper* than today.

Two secondary findings from the same runs, both actionable:

- **A plain instance method is 41 ns FASTER than a classmethod** (48.7 vs 89.7). Moving the hot path
  to instance methods is a small performance *win*, not a cost.
- **`*args, **kwargs` packing dominates the closure mechanisms.** H (a fixed-signature lambda) is
  99.2 ns; I (a `*a, **kw` closure) is 280.2 ns, on otherwise identical work. Any closure introduced
  on this path must take a fixed signature.

### 4.1 A trap, found by measurement rather than by reading

The obvious metaclass formulation — put `debug` on the metaclass for class access and on the class
for instance access — **does not work**, and fails in a way that reaches the caller as an argument
shift rather than an error at definition time:

```text
TypeError: MetaStyle.debug() missing 1 required positional argument: 'message'
```

`type.__getattribute__` searches the **class's own MRO before** falling back to non-data metaclass
attributes, so the class's instance method shadows the metaclass's function for `Logger.debug`, and
the class call arrives with `self=1, level="m"`. Only a **data** descriptor on the metaclass (one
defining `__set__`/`__delete__`) outranks the class `__dict__` — and that is row **F**, at +263.7 ns.

This is worth recording because the shadowing is silent at import and at definition: nothing warns,
and a suite that only exercises the instance path stays green.

---

## 5. Recommendation — two objects, not one overloaded name

**Do not overload a single name to dispatch on its receiver.** The measured cost is 3–6× on the path
this whole arc has been making cheaper, and it buys nothing that two bindings do not already give.

Recommended shape:

1. **`Logger` keeps its 32 classmethods, unchanged.** The 109 `Logger.M(...)` sites — 64 of them
   inside `logger.py` — are untouched, at 89.7 ns. The class path remains the process-wide default
   and the bootstrap path (it must keep working before any instance exists; `logger.py`'s own
   construction logging depends on that).
2. **Add a factory returning a per-logger instance** whose class defines the eight emit methods and
   `isEnabledFor` as **plain instance methods** — 48.7 ns, *faster than today*, with the level knobs
   as instance attributes falling back to the class value when unset.
3. **The 967 `self.logger.M(...)` sites change zero characters.** Only the **14 bind sites** change,
   from `self.logger = Logger` to `self.logger = <factory>("candidate_unit")` — plus the two indirect
   binds at `cascade_correlation.py:667` and `spiral_problem.py:310`.

This delivers precisely what the owner asked for — one fully-functional logging path reachable both
as class calls and as per-instance calls, with per-call-site tuning — **and it is faster on the hot
path than the status quo**, because the 967-site majority moves from classmethod dispatch to instance
dispatch.

> **Naming note.** `get_logger` is **taken twice** and cannot be the factory name without a
> collision: `Logger.get_logger(self)` (`logger.py:1281`) is an instance method that returns `self`,
> and `LogConfig.get_logger(self)` (`log_config.py:493`) returns `self.logger`. Pick a new name
> (`Logger.for_name(...)` / `Logger.bind(...)` / `Logger.sub(...)`) or deliberately re-purpose
> `logger.py:1281`, whose body is `return self`.

---

## 6. Issues, risks and concerns of a dual-access design

Answering the owner's question directly. These apply to the recommendation in §5 as much as to the
rejected mechanisms, except where noted.

| # | concern | severity | notes |
|---|---------|----------|-------|
| 1 | **Two sources of truth for "the level".** A class-level default and per-instance overrides mean a record's fate depends on which binding the call site used. An operator who sets the global level and sees a module stay quiet has no error to read. | **high** | Mitigate with P4-G1's precedence truth table and a `Logger.describe_levels()`-style dump. This is the cost of the feature, not of the mechanism. |
| 2 | **The state split in N-3 gets WIDER before it gets narrower.** Today `isEnabledFor` reads `_log_level` and the emit filter reads `_level_logger_config`/`_level_logger_name`, and they disagree. Adding a per-instance layer over an already-disjoint pair multiplies the states. | **high** | **P1.1 is a hard prerequisite.** Do not start P4 until one resolved value drives both paths. ROADMAP already gates P4 on P1 for a weaker reason than this. |
| 3 | **Bootstrap ordering.** Class-level logging must work before any instance exists — `logger.py` logs during its own construction, and `logger.py:768`'s root-clobber guard runs at `Logger.__init__` time. A factory that itself logs through an instance can recurse. | medium | The factory must use the class path only, and must not be reachable from `Logger.__init__`. |
| 4 | **Forkserver inheritance of per-instance state.** `cascade_correlation` is preloaded (RECON N-2), so instances created at import or forkserver time are inherited by every child. Per-instance levels set in the parent **after** forkserver start are invisible to children — the classic P4.4 failure. | medium | P4.4 already requires demonstration in a child; extend it to per-instance state specifically. |
| 5 | **Pickling.** `CandidateUnit.__setstate__` already re-binds `self.logger` (`conftest.py:914` stubs exactly this seam), so a logger instance must be excluded from `__getstate__` or be trivially reconstructible. A closure-bearing instance (mechanism **H**) is **not picklable**. | medium | Argues against **H** and for plain instance methods, which pickle normally. |
| 6 | **Reference cycles.** Mechanisms **H** and **I** put a closure over `self` into `self`, creating an `instance → closure → instance` cycle that only the cycle collector reclaims. At one logger per `CandidateUnit`, across a cap-64 pool, that is real. | medium | Another argument for plain instance methods. Not applicable to §5's recommendation. |
| 7 | **Memory per instance.** A logger per `CandidateUnit` costs one object plus its level attributes per candidate. Small, but it is per-candidate in a pool that reaches 64. | low | Measure in P4; keep the instance's `__slots__` tight. |
| 8 | **The mirror.** `candidate_unit/`, `utils/`, `log_config/` and `cascor_constants/` are byte-gated (ROADMAP trap 2), and `log_config/logger/logger.py` is on `_INTENTIONAL_DIVERGENCE`. Any P4 edit touching the bind sites in `candidate_unit.py` carries a mirror re-extraction **and a package release**. | medium | Now governed by the ruled decision 4 (converge — see §8). |
| 9 | **Two idioms invite drift.** Once both work, new code picks either, and the choice stops being deliberate. | low | P6.3's lint rule is the natural home for a "use the instance binding inside `candidate_unit`/`cascade_correlation`" rule. |
| 10 | **`isEnabledFor` as an instance method changes the 8 guard sites' meaning.** They currently ask a class-wide question; they would ask a per-instance one. That is the intent, but it is a behaviour change at exactly the 8 sites whose integers are already wrong (RECON N-3). | medium | Sequence P1.3 (fix the integers) **before** P4 changes what the predicate means. |

**Not a concern, contrary to expectation:** raw dispatch cost, *provided* §5's shape is used. The
967-site majority gets **faster**. It is only the one-name-two-receivers mechanisms that are
expensive, and none of them is needed.

---

## 7. What this document does not settle

- **It does not measure the real logger.** All figures in §4 are dispatch-only, on synthetic bodies,
  on one machine and one interpreter (Python 3.13.13, `JuniperCascor1`). They size the *delta between
  mechanisms*, which is the decision at hand; they are **not** a claim about cascor's logging share.
  Run-to-run noise reached 57 % on the fastest rows, which is why minima are reported.
- **It does not re-open P1.** The state split (RECON N-3) and the `is_valid_level` typo
  (`logger.py:341`, N-3a) remain prerequisites, and concern 2 above raises their priority.
- **It does not decide the precedence chain.** ROADMAP P4.2's chain — per-logger env → global env →
  per-logger config → global config → default — is unchanged by this analysis.
- **It carries no post-#598 measurement.** P0.1 still owes the corpus; nothing here substitutes.

---

## 8. Decisions already ruled, recorded here for the implementer

Taken by the owner on 2026-09-09 against ROADMAP §13; §13.1 of that document is the canonical record.

| # | decision | ruling |
|---|----------|--------|
| 1 | Does P2 run at all? | **Pre-authorised: drop P2 if P0.2 puts the logging share below 10 %** |
| 2 | P3.2 console-sink default | **On by default, off in the harness profile** — a policy switch, not runtime detection |
| 3 | P5.1 `configure_logging` fork | **Narrow the fork to the delta**; the rotator stays |
| 4 | P0.6 mirror policy | **Converge** — re-extract `logger.py`, retire `_INTENTIONAL_DIVERGENCE` |
| 5 | P4.1 per-logger shape | **open** — this document is the input; see §9 |
| 6 | Call-site migration scope | **P6.1 + P6.2 + P6.3 authorised; P6.4 open** pending the sample review |
| 7 | §7.1 swallowed-pytest investigation | **Authorised**, as written in DESIGN §7.1 |

Decision 4's ruling matters to this document: it means a P4 edit to `candidate_unit.py`'s bind sites
carries a live mirror obligation, and that the package's logger — currently pre-#563 **and**
pre-#598, published on PyPI at 0.1.0 — is being brought back into line rather than left frozen.

---

## 9. Decision required, and what follows it

**The question for the owner:** adopt §5's shape — `Logger` keeps its classmethods, a factory returns
per-logger instances with plain instance methods, 967 call sites unchanged, 14 binds edited?

The alternatives remain on the table and are cheap to state:

- **(a) §5's shape — two bindings.** Recommended. M-sized. Faster on the hot path.
- **(b) Named sub-loggers on the class only.** A name-keyed level map consulted by the existing
  classmethods; no instances, no bind changes at all. Smallest change, but every call site must then
  *pass* its name, which is a 1,188-site edit — the opposite of the trade in (a).
- **(c) One overloaded name, dispatching on receiver.** Rejected on measurement: +190 to +438 ns per
  call, 3–6× the status quo, on the 646 k-call path.

**Once selected, this document is extended** with the implementation design: the concrete class
shape, the precedence chain wired to P4.2, the `__getstate__`/`__setstate__` contract from concern 5,
the bootstrap ordering rule from concern 3, the forkserver demonstration from concern 4, and the
per-step mirror obligations under the ruled decision 4.

---

## 10. References

- [cascor#573](https://github.com/pcalnon/juniper-cascor/issues/573) — the issue and owner scope
- **ROADMAP** §13 decision 5, §7 (Phase 4), P4-G1…P4-G4
- **RECON** N-3 / N-3a (the level-state split), N-2 (forkserver), §3.1 (the call-site census method)
- **DESIGN** §7 decisions 1–6, and §7.1 (the swallowed-pytest protocol)
- `juniper-ml/util/ad-hoc/2026-09-09_p41_logger_binding_census.py` — the §2 census
- `juniper-ml/util/ad-hoc/2026-09-09_p41_dualpath_mechanisms_bench.py` — the §4 table
- `juniper-ml/util/ad-hoc/2026-09-09_p41_hybrid_descriptor_bench.py` — the naive-descriptor rows
- `juniper-ml/util/ad-hoc/2026-09-02_logging_doc_refutation_probe.py` — the level-state probe
