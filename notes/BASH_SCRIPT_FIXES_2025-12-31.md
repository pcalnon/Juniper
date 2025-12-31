# Bash Script Infrastructure Fixes - 2025-12-31

## Overview

This document details additional fixes applied to the Juniper Canopy bash script infrastructure following the initial analysis on 2025-12-30. These fixes resolved remaining issues that prevented `get_code_stats.bash` from running successfully.

## Additional Fixes Applied

### 1. Fixed `ps` Command Output Pollution

**Problem:** The `ps -o args="${PARENT_PID}"` command on Linux returned the entire process list instead of just the parent process args, polluting the parent process detection.

**File:** `conf/common.conf`

**Fix:** Changed to use `-p` flag properly:

```bash
# Before:
export PARENT_CALL="$(ps -o args="${PARENT_PID}")"

# After:
export PARENT_CALL="$(ps -p "${PARENT_PID}" -o args= 2>/dev/null | head -1)"
```

---

### 2. Fixed Missing Shift in get_file_todo.bash

**Problem:** The while loop in argument parsing didn't shift at the end of the case statement, causing an infinite loop.

**File:** `util/get_file_todo.bash`

**Fix:** Added `shift` after `esac`:

```bash
    esac
    shift
done
```

---

### 3. Fixed SIZE_LABELS Array Syntax Error

**Problem:** `${SIZE_LABELS[* ]}` had an extra space causing a syntax error.

**File:** `conf/get_code_stats.conf`

**Fix:** Changed to `${SIZE_LABELS[*]}` (removed space).

---

### 4. Exported Missing Logging Helper Functions

**Problem:** The `evaluate_log_level`, `logger`, `debug_flag_matters`, and `display_level_matters` functions were used by exported log_* functions but were not themselves exported, causing "command not found" errors in subshells.

**File:** `conf/logging_functions.conf`

**Fix:** Added exports after function definitions:

```bash
export -f logger
export -f debug_flag_matters
export -f display_level_matters
export -f evaluate_log_level
```

---

### 5. Fixed Logging to stdout Instead of stderr

**Problem:** Logging output was going to stdout via `tee`, polluting the output of scripts that capture stdout.

**Files:** `conf/logging_functions.conf`, `conf/init.conf`

**Fix:** Changed from `2>&1` to `>&2`:

```bash
# Before:
printf ... | tee -a "${LOG_FILE}" 2>&1

# After:
printf ... | tee -a "${LOG_FILE}" >&2
```

---

### 6. Fixed log_critical Called Before Definition

**Problem:** Line 124 of `common.conf` used `log_critical` but logging.conf (which defines it) wasn't sourced yet.

**File:** `conf/common.conf`

**Fix:** Changed to use `log_fatal` which is defined earlier in `init.conf`.

---

### 7. Removed Exported Guards from Config Files

**Problem:** Config file re-source guards were exported, causing subprocesses to skip their own configuration chain when called from a parent script that had already sourced the configs.

**Files:** Multiple config files including:

- `conf/common.conf`
- `conf/logging.conf`
- `conf/logging_functions.conf`
- `conf/logging_colors.conf`
- `conf/common_functions.conf`
- `conf/get_code_stats.conf`
- `conf/get_module_filenames.conf`
- `conf/get_file_todo.conf`
- `conf/get_file_todo_functions.conf`

**Fix:** Removed `export` keyword from guard variable assignments so each subprocess gets its own config chain:

```bash
# Before:
export COMMON_CONF_SOURCED="${TRUE}"

# After:
COMMON_CONF_SOURCED="${TRUE}"
```

---

### 8. Performance Optimization - Inlined Find and Grep

**Problem:** The script was extremely slow because it called subscripts (which source the entire config chain) for every file iteration.

**File:** `util/get_code_stats.bash`

**Fix:** Replaced subprocess calls with inline commands:

```bash
# Inlined file discovery (was: ${GET_FILENAMES_SCRIPT} ${FILENAMES_SCRIPT_PARAMS})
while read -r i; do
    ...
done <<< "$(find "${SRC_DIR}" \( -name "*.py" ! -name "*__init__*" ! -name "*test_*.py" \) -type f 2>/dev/null)"

# Inlined TODO counting (was: ${GET_FILE_TODO_SCRIPT} ...)
CURRENT_TODOS="$(grep -ic "${SEARCH_TERM_DEFAULT}" "${FILE_PATH}" 2>/dev/null || echo "0")"
```

---

### 9. Fixed Method Count Grep Missing -c Flag

**Problem:** The grep command for counting methods was missing the `-c` flag, so it returned matching lines instead of a count.

**File:** `util/get_code_stats.bash`

**Fix:** Added `-c` flag to grep command:

```bash
CURRENT_METHODS=$(grep -c ${FIND_METHOD_PARAMS} "${FIND_METHOD_REGEX}" "${FILE_PATH}" 2>/dev/null || echo "0")
```

---

## Summary

With these fixes, `get_code_stats.bash` now runs successfully and produces correct output:

```bash
Display Stats for the JuniperCanopy Project

Filename                        Lines   Methods   TODOs      Size
----------------------------   ------  --------  ------    ------
websocket_manager.py              761        10       1     28 KB
config_manager.py                 492        19       1     20 KB
...

Project JuniperCanopy Summary:

Total Files:   23
Total Methods: 86
Total Lines:   13468
Total TODOs:   25
Total Size:    556 KB
```

---

## Files Modified in This Session

### Configuration Files

- `conf/common.conf` - ps command fix, log_fatal usage, removed export from guard
- `conf/logging_functions.conf` - Export helper functions, stderr redirect
- `conf/init.conf` - stderr redirect for log_fatal
- `conf/get_code_stats.conf` - SIZE_LABELS syntax fix
- `conf/get_file_todo_functions.conf` - Removed export from guard

### Script Files

- `util/get_code_stats.bash` - Inlined find/grep, method count fix
- `util/get_file_todo.bash` - Missing shift fix
