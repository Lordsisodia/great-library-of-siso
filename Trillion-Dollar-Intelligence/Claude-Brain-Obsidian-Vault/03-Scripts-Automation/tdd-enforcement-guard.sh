#!/bin/bash

# TDD Enforcement Guard - Inspired by nizos/tdd-guard
FILE_PATH="$1"
OPERATION="$2"  # edit, write, or check
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# TDD settings
TDD_ENABLED="${CLAUDE_TDD_ENABLED:-true}"
TDD_MODE="${CLAUDE_TDD_MODE:-strict}"  # strict, relaxed, monitor
TDD_BYPASS="${CLAUDE_TDD_BYPASS:-false}"

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/tdd-guard.log; }

# Check if TDD enforcement is enabled
if [[ "$TDD_ENABLED" != "true" ]] || [[ "$TDD_BYPASS" == "true" ]]; then
    exit 0
fi

# Detect if file is a test file
is_test_file() {
    local file="$1"
    case "$file" in
        *.test.ts|*.test.tsx|*.test.js|*.test.jsx) return 0 ;;
        *.spec.ts|*.spec.tsx|*.spec.js|*.spec.jsx) return 0 ;;
        */__tests__/*) return 0 ;;
        */test/*) return 0 ;;
        */tests/*) return 0 ;;
        *) return 1 ;;
    esac
}

# Find corresponding test file
find_test_file() {
    local source_file="$1"
    local dir=$(dirname "$source_file")
    local base=$(basename "$source_file" | sed 's/\.[^.]*$//')
    local ext="${source_file##*.}"
    
    # Common test file patterns
    local test_patterns=(
        "$dir/$base.test.$ext"
        "$dir/$base.spec.$ext"
        "$dir/__tests__/$base.test.$ext"
        "$dir/__tests__/$base.spec.$ext"
        "${dir/src/test}/$base.test.$ext"
        "${dir/src/tests}/$base.test.$ext"
    )
    
    for pattern in "${test_patterns[@]}"; do
        if [[ -f "$pattern" ]]; then
            echo "$pattern"
            return 0
        fi
    done
    
    return 1
}

# Find corresponding source file
find_source_file() {
    local test_file="$1"
    local dir=$(dirname "$test_file")
    local base=$(basename "$test_file" | sed 's/\.(test|spec)\.[^.]*$//')
    local ext="${test_file##*.}"
    
    # Remove test-specific parts from extension
    ext=$(echo "$ext" | sed 's/test\.//' | sed 's/spec\.//')
    
    # Common source file patterns
    local source_patterns=(
        "$dir/$base.$ext"
        "${dir/__tests__/}/$base.$ext"
        "${dir/test/src}/$base.$ext"
        "${dir/tests/src}/$base.$ext"
    )
    
    for pattern in "${source_patterns[@]}"; do
        if [[ -f "$pattern" ]]; then
            echo "$pattern"
            return 0
        fi
    done
    
    return 1
}

# Check if tests exist and are passing
check_test_status() {
    local source_file="$1"
    local test_file=$(find_test_file "$source_file")
    
    if [[ -z "$test_file" ]]; then
        return 1  # No test file found
    fi
    
    # Run tests for this specific file
    local project_dir=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    
    cd "$project_dir"
    
    # Detect test framework and run appropriate command
    if [[ -f "package.json" ]]; then
        if grep -q "vitest" package.json; then
            npx vitest run "$test_file" --reporter=silent >/dev/null 2>&1
        elif grep -q "jest" package.json; then
            npx jest "$test_file" --silent >/dev/null 2>&1
        else
            npm test -- "$test_file" >/dev/null 2>&1
        fi
    else
        return 2  # Cannot determine test framework
    fi
}

# TDD violation messages
show_tdd_violation() {
    local violation_type="$1"
    local file="$2"
    
    case "$violation_type" in
        "no_test")
            echo "🚨 TDD VIOLATION: No test file found for $file"
            echo "🧪 TDD requires tests to exist before implementation"
            echo "💡 Create test file first, then implement functionality"
            echo "🔧 Suggested test file: $(dirname "$file")/$(basename "$file" .${file##*.}).test.${file##*.}"
            ;;
        "failing_tests")
            echo "🚨 TDD VIOLATION: Tests are failing for $file"
            echo "🔴 RED phase detected - tests should fail first"
            echo "💡 This might be correct if you're in the RED phase"
            echo "✅ Implement minimal code to make tests pass (GREEN phase)"
            ;;
        "passing_tests")
            echo "🚨 TDD VIOLATION: Tests are already passing"
            echo "✅ GREEN phase detected - no implementation needed"
            echo "💡 Consider refactoring (REFACTOR phase) or adding new failing tests"
            ;;
    esac
}

# TDD mode enforcement
enforce_tdd() {
    local file="$1"
    local operation="$2"
    
    if is_test_file "$file"; then
        log "🧪 Test file modification allowed: $file"
        return 0
    fi
    
    # Check for corresponding test file
    local test_file=$(find_test_file "$file")
    
    if [[ -z "$test_file" ]]; then
        case "$TDD_MODE" in
            "strict")
                show_tdd_violation "no_test" "$file"
                log "❌ TDD STRICT: Blocked modification of $file (no tests)"
                return 1
                ;;
            "relaxed")
                echo "⚠️ TDD WARNING: No test file found for $file"
                echo "💡 Consider creating tests for better code quality"
                log "⚠️ TDD RELAXED: Warning for $file (no tests)"
                return 0
                ;;
            "monitor")
                log "📊 TDD MONITOR: No tests for $file"
                return 0
                ;;
        esac
    fi
    
    # Check test status
    local test_status
    check_test_status "$file"
    test_status=$?
    
    case "$test_status" in
        0)  # Tests passing
            if [[ "$operation" == "write" ]]; then
                show_tdd_violation "passing_tests" "$file"
                log "⚠️ TDD: Tests already passing for $file"
            fi
            ;;
        1)  # Tests failing (good for TDD)
            log "✅ TDD: Tests failing for $file (RED phase)"
            ;;
        2)  # Cannot run tests
            log "❓ TDD: Cannot determine test status for $file"
            ;;
    esac
    
    return 0
}

# Generate TDD suggestions
suggest_tdd_workflow() {
    local file="$1"
    
    echo ""
    echo "🔄 TDD WORKFLOW SUGGESTIONS:"
    echo "1. 🔴 RED: Write failing test first"
    echo "2. 🟢 GREEN: Write minimal code to pass"
    echo "3. 🔧 REFACTOR: Improve code while keeping tests green"
    echo ""
    
    if ! is_test_file "$file"; then
        local test_file=$(find_test_file "$file")
        if [[ -z "$test_file" ]]; then
            echo "💡 Next step: Create test file for $file"
            echo "🔧 Command: touch $(dirname "$file")/$(basename "$file" .${file##*.}).test.${file##*.}"
        else
            echo "💡 Test file exists: $test_file"
            echo "🔧 Consider running: npm test $test_file"
        fi
    fi
}

# Main TDD enforcement
main() {
    if [[ -z "$FILE_PATH" ]]; then
        log "❌ No file path provided"
        exit 1
    fi
    
    log "🔍 TDD Check: $FILE_PATH ($OPERATION mode: $TDD_MODE)"
    
    if enforce_tdd "$FILE_PATH" "$OPERATION"; then
        if [[ "$TDD_MODE" == "strict" ]] || [[ "$TDD_MODE" == "relaxed" ]]; then
            suggest_tdd_workflow "$FILE_PATH"
        fi
        exit 0
    else
        log "❌ TDD enforcement blocked modification"
        echo ""
        echo "🛑 TDD ENFORCEMENT: File modification blocked"
        echo "🧪 To bypass: CLAUDE_TDD_BYPASS=true or CLAUDE_TDD_ENABLED=false"
        echo "⚙️ To change mode: CLAUDE_TDD_MODE=relaxed|monitor"
        exit 1
    fi
}

# Initialize
mkdir -p ~/.claude/logs

# Execute
main