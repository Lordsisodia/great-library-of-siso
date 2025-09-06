#!/bin/bash

# 10x Auto Test Runner - Smart test execution and coverage
TEST_FILE="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/hooks.log; }

log "🧪 Auto test runner for $TEST_FILE"

if [[ -f "$TEST_FILE" ]]; then
    project_dir=$(dirname "$TEST_FILE")
    cd "$project_dir"
    
    # Find test command
    test_cmd=""
    if [[ -f "package.json" ]]; then
        if grep -q '"test".*vitest' package.json; then
            test_cmd="npx vitest run"
        elif grep -q '"test".*jest' package.json; then
            test_cmd="npx jest"
        elif grep -q '"test"' package.json; then
            test_cmd="npm test"
        fi
    fi
    
    if [[ -n "$test_cmd" ]]; then
        log "🎯 Running tests: $test_cmd"
        
        # Run specific test file
        rel_test=$(basename "$TEST_FILE")
        $test_cmd "$rel_test" --watchAll=false --silent 2>/dev/null || log "❌ Tests failed"
        
        # Run related tests (find source file)
        source_file="${TEST_FILE%.test.*}.${TEST_FILE##*.test.}"
        if [[ -f "$source_file" ]]; then
            log "🔗 Running tests for related source file"
            $test_cmd "$source_file" --watchAll=false --silent 2>/dev/null || true
        fi
        
        # Generate coverage for this file
        if command -v npx >/dev/null; then
            log "📊 Generating coverage report"
            $test_cmd --coverage --coverageReporters=text-summary 2>/dev/null | tail -5 >> ~/.claude/logs/test-coverage.log || true
        fi
        
        # Smart test insights
        if [[ -f "coverage/lcov.info" ]]; then
            coverage_percent=$(grep -o 'SF:.*' coverage/lcov.info | wc -l 2>/dev/null || echo "0")
            log "📈 Coverage data updated: $coverage_percent files analyzed"
        fi
        
    else
        log "⚠️ No test runner detected in package.json"
    fi
    
    # Check test file quality
    if grep -q "describe\|it\|test" "$TEST_FILE"; then
        test_count=$(grep -c "it\|test" "$TEST_FILE")
        log "✅ Test file contains $test_count test cases"
    fi
fi

log "🏁 Auto test execution complete"