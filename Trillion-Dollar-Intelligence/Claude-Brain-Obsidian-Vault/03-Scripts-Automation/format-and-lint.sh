#!/bin/bash

# SISO Ecosystem - Auto Format and Lint Hook
# Automatically formats and lints code files when Claude edits them

FILE_PATHS="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Log the operation
echo "[$TIMESTAMP] Claude hook: Formatting and linting $FILE_PATHS" >> ~/.claude/logs/hooks.log

# Function to run commands safely with error handling
run_safe() {
    local cmd="$1"
    local file="$2"
    
    if command -v $(echo $cmd | cut -d' ' -f1) >/dev/null 2>&1; then
        eval "$cmd \"$file\"" 2>/dev/null || true
    fi
}

# Process each file path
for file in $FILE_PATHS; do
    if [[ -f "$file" ]]; then
        case "$file" in
            *.ts|*.tsx|*.js|*.jsx)
                # Check if we're in a project with prettier/eslint
                if [[ -f "$(dirname "$file")/package.json" ]] || [[ -f "$(pwd)/package.json" ]]; then
                    # Try prettier first
                    run_safe "npx prettier --write" "$file"
                    
                    # Try eslint with fix
                    run_safe "npx eslint --fix" "$file"
                fi
                ;;
        esac
        
        echo "[$TIMESTAMP] Processed: $file" >> ~/.claude/logs/hooks.log
    fi
done

exit 0