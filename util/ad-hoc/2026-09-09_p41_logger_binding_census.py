#!/usr/bin/env python3
"""
Project:     Juniper
Sub-Project: juniper-ml (cascor#573 logging redesign, ROADMAP decision 5 / P4.1)
Application: ad-hoc analysis
Author:      Paul Calnon
Version:     0.1.0
License:     MIT License

Purpose: census the ACCESS SYNTAX of every live Path-A logger call site in
juniper-cascor, to size what a move to per-instance loggers would actually break.

ROADMAP 13 decision 5 states per-instance "breaks ~1,200 call sites". That counts
call sites. This script separates them by how the logger is REACHED, because a site
written ``self.logger.debug(...)`` is syntactically identical whether ``self.logger``
holds the Logger class or a Logger instance -- only the BIND site changes. The sites
that genuinely break are those naming the class directly.

Population, per RECON 3.1: tracked src/, less cascade_correlation/backups/ (dead),
less src/api/ (stdlib-bound, not Path A), less src/tests/.

Read-only.
"""
import re
import subprocess
import sys

CASCOR = "/home/pcalnon/Development/python/Juniper/juniper-cascor"
M = r"(?:trace|verbose|debug|info|warning|error|critical|fatal|isEnabledFor)"

PATTERNS = {
    "self.logger.M()  -- instance-style, survives either design": re.compile(r"\bself\.logger\." + M + r"\("),
    "cls.logger.M()   -- class-attr style": re.compile(r"\bcls\.logger\." + M + r"\("),
    "bare logger.M()  -- local name, survives if the local binds an instance": re.compile(r"(?<![\w.])logger\." + M + r"\("),
    "Logger.M()       -- DIRECT CLASS ACCESS, breaks under instance methods": re.compile(r"(?<![\w.])Logger\." + M + r"\("),
}

BINDS = {
    "self.logger = Logger": re.compile(r"\bself\.logger\s*=\s*Logger\b(?!\()"),
    "logger = Logger (local/module)": re.compile(r"(?<![\w.])logger\s*=\s*Logger\b(?!\()"),
    "= Logger() (already an instance)": re.compile(r"=\s*Logger\("),
}

EXCLUDES = (
    ":!src/cascade_correlation/backups",
    ":!src/api",
    ":!src/tests",
)


def live_files():
    # NOTE: git grep -E is POSIX ERE -- it does NOT support Python's (?:...) group.
    # Keep this discovery pattern plain; the Python regexes above do the real work.
    out = subprocess.run(
        ["git", "-C", CASCOR, "grep", "-l", "-E", r"[Ll]ogger", "HEAD", "--", "src", *EXCLUDES],
        capture_output=True, text=True, check=False).stdout
    return [ln.split(":", 1)[1] for ln in out.splitlines() if ":" in ln]


def main():
    call_tot = dict.fromkeys(PATTERNS, 0)
    bind_tot = dict.fromkeys(BINDS, 0)
    call_hits = {k: [] for k in PATTERNS}
    bind_hits = {k: [] for k in BINDS}
    per_file_direct = {}

    for path in live_files():
        blob = subprocess.run(["git", "-C", CASCOR, "show", f"HEAD:{path}"],
                              capture_output=True, text=True, check=False).stdout
        for n, line in enumerate(blob.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for key, rx in PATTERNS.items():
                c = len(rx.findall(line))
                if c:
                    call_tot[key] += c
                    if key.startswith("Logger."):
                        per_file_direct[path] = per_file_direct.get(path, 0) + c
                    if len(call_hits[key]) < 6:
                        call_hits[key].append(f"{path}:{n}: {line.strip()[:150]}")
            for key, rx in BINDS.items():
                c = len(rx.findall(line))
                if c:
                    bind_tot[key] += c
                    if len(bind_hits[key]) < 20:
                        bind_hits[key].append(f"{path}:{n}: {line.strip()[:120]}")

    print("=== CALL SITES, by access syntax ===")
    for k, v in sorted(call_tot.items(), key=lambda kv: -kv[1]):
        print(f"{v:>5}  {k}")
    total = sum(call_tot.values())
    breaks = call_tot["Logger.M()       -- DIRECT CLASS ACCESS, breaks under instance methods"]
    print(f"{'-' * 74}")
    print(f"{total:>5}  TOTAL live Path-A call sites")
    print(f"{breaks:>5}  would need editing under a pure instance-method design ({100 * breaks / total:.1f}%)")

    print("\n=== BIND SITES (where `logger` gets its value) ===")
    for k, v in sorted(bind_tot.items(), key=lambda kv: -kv[1]):
        print(f"{v:>5}  {k}")

    print("\n=== DIRECT CLASS-ACCESS sites, by file ===")
    for p, c in sorted(per_file_direct.items(), key=lambda kv: -kv[1]):
        print(f"{c:>5}  {p}")

    print("\n=== SAMPLES ===")
    for k in PATTERNS:
        if not call_hits[k]:
            continue
        print(f"\n--- {k} ---")
        for h in call_hits[k]:
            print(f"   {h}")

    print("\n=== ALL BIND SITES ===")
    for k in BINDS:
        print(f"\n--- {k} ({bind_tot[k]}) ---")
        for h in bind_hits[k]:
            print(f"   {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
