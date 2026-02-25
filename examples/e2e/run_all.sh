#!/usr/bin/env bash
# Master runner for kraang end-to-end tests.
# Usage: ./examples/e2e/run_all.sh [--keep-on-fail]
set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
KEEP_ON_FAIL=false
for arg in "$@"; do
    case "$arg" in
        --keep-on-fail) KEEP_ON_FAIL=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate project root (directory containing pyproject.toml)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo -e "${RED}ERROR: pyproject.toml not found at $PROJECT_ROOT${RESET}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Temp directory + cleanup trap
# ---------------------------------------------------------------------------
TMPDIR_BASE="$(mktemp -d)"
FAILED=0

cleanup() {
    if [[ "$FAILED" -ne 0 && "$KEEP_ON_FAIL" == true ]]; then
        echo -e "\n${YELLOW}--keep-on-fail: temp dir preserved at $TMPDIR_BASE${RESET}"
    else
        rm -rf "$TMPDIR_BASE"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Create venv & install kraang
# ---------------------------------------------------------------------------
echo -e "${BOLD}Setting up virtual environment...${RESET}"
python3 -m venv "$TMPDIR_BASE/venv"
# shellcheck disable=SC1091
source "$TMPDIR_BASE/venv/bin/activate"

echo "Installing kraang (editable)..."
pip install --quiet -e "$PROJECT_ROOT[dev]"

# ---------------------------------------------------------------------------
# Discover and run e2e tests
# ---------------------------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0
TOTAL=0
RESULTS=()

for test_file in "$SCRIPT_DIR"/e2e_*.py; do
    [[ -f "$test_file" ]] || continue
    test_name="$(basename "$test_file")"
    TOTAL=$((TOTAL + 1))

    echo -e "\n${BOLD}Running $test_name...${RESET}"
    start_time=$(date +%s)

    set +e
    python "$test_file"
    exit_code=$?
    set -e

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    if [[ $exit_code -eq 0 ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("${GREEN}PASS${RESET}  $test_name  (${elapsed}s)")
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED=1
        RESULTS+=("${RED}FAIL${RESET}  $test_name  (${elapsed}s)")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}========================================${RESET}"
echo -e "${BOLD}  E2E Test Summary${RESET}"
echo -e "${BOLD}========================================${RESET}"

for line in "${RESULTS[@]}"; do
    echo -e "  $line"
done

echo -e "${BOLD}----------------------------------------${RESET}"
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "  ${GREEN}All $TOTAL test file(s) passed.${RESET}"
else
    echo -e "  ${RED}$FAIL_COUNT/$TOTAL test file(s) failed.${RESET}"
fi
echo -e "${BOLD}========================================${RESET}"

exit $((FAIL_COUNT > 0 ? 1 : 0))
