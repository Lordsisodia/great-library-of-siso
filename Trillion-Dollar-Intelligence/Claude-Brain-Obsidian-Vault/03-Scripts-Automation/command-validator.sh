#!/bin/bash

# Command Validator - Inspired by shuntaka9576/blocc
COMMANDS=("$@")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Results tracking
declare -a results=()
failed_count=0

# Execute command with detailed reporting
execute_command() {
    local cmd="$1"
    local temp_stdout=$(mktemp)
    local temp_stderr=$(mktemp)
    
    echo "🔧 Executing: $cmd" >&2
    
    # Execute command and capture output
    bash -c "$cmd" > "$temp_stdout" 2> "$temp_stderr"
    local exit_code=$?
    
    local stdout_content=$(cat "$temp_stdout")
    local stderr_content=$(cat "$temp_stderr")
    
    # Create result object
    local result="{
        \"command\": \"$cmd\",
        \"exitCode\": $exit_code,
        \"stdout\": \"$(echo "$stdout_content" | sed 's/"/\\"/g')\",
        \"stderr\": \"$(echo "$stderr_content" | sed 's/"/\\"/g')\"
    }"
    
    results+=("$result")
    
    if [[ $exit_code -ne 0 ]]; then
        failed_count=$((failed_count + 1))
        echo "❌ FAILED: $cmd (exit code: $exit_code)" >&2
        if [[ -n "$stderr_content" ]]; then
            echo "Error: $stderr_content" >&2
        fi
    else
        echo "✅ PASSED: $cmd" >&2
    fi
    
    # Cleanup
    rm -f "$temp_stdout" "$temp_stderr"
    
    return $exit_code
}

# Generate JSON error report
generate_error_report() {
    local message="$failed_count command(s) failed"
    
    echo "{"
    echo "  \"message\": \"$message\","
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"results\": ["
    
    for i in "${!results[@]}"; do
        echo "    ${results[$i]}"
        if [[ $i -lt $((${#results[@]} - 1)) ]]; then
            echo ","
        fi
    done
    
    echo "  ]"
    echo "}"
}

# Smart command suggestions
suggest_fixes() {
    for cmd in "${COMMANDS[@]}"; do
        case "$cmd" in
            *"npm run lint"*)
                echo "💡 Lint failed: Try 'npm run lint -- --fix' for auto-fixes" >&2
                ;;
            *"npm run build"*)
                echo "💡 Build failed: Check TypeScript errors with 'npx tsc --noEmit'" >&2
                ;;
            *"npm test"*)
                echo "💡 Tests failed: Run 'npm test -- --verbose' for detailed output" >&2
                ;;
            *"tsc"*)
                echo "💡 TypeScript failed: Check your types and imports" >&2
                ;;
        esac
    done
}

# Main execution
main() {
    echo "🚀 Command validation started" >&2
    echo "Commands to execute: ${#COMMANDS[@]}" >&2
    
    # Execute all commands
    for cmd in "${COMMANDS[@]}"; do
        execute_command "$cmd"
    done
    
    # Report results
    if [[ $failed_count -gt 0 ]]; then
        echo "" >&2
        echo "❌ Validation failed: $failed_count command(s) failed" >&2
        suggest_fixes
        echo "" >&2
        
        # Generate JSON report
        generate_error_report
        
        # Exit with code 2 (community standard for blocking)
        exit 2
    else
        echo "" >&2
        echo "✅ All commands passed validation" >&2
        echo "{\"message\": \"All commands passed\", \"timestamp\": \"$TIMESTAMP\"}"
        exit 0
    fi
}

# Check if commands provided
if [[ ${#COMMANDS[@]} -eq 0 ]]; then
    echo "Usage: $0 \"command1\" \"command2\" ..." >&2
    echo "Example: $0 \"npm run lint\" \"npm run build\" \"npm test\"" >&2
    exit 1
fi

# Execute
main