#!/bin/bash

# 10x Smart Code Processor - Auto-format, lint, test, and optimize
FILE_PATHS="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/hooks.log; }

# Smart project detection
detect_project() {
    local file="$1"
    local dir="$(dirname "$file")"
    
    # Find nearest package.json
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/package.json" ]]; then
            echo "$dir"
            return
        fi
        dir="$(dirname "$dir")"
    done
    echo "$(pwd)"
}

run_in_project() {
    local cmd="$1"
    local project_dir="$2"
    (cd "$project_dir" && eval "$cmd" 2>/dev/null) || true
}

for file in $FILE_PATHS; do
    if [[ -f "$file" ]]; then
        project_dir=$(detect_project "$file")
        rel_file=$(realpath --relative-to="$project_dir" "$file" 2>/dev/null || basename "$file")
        
        log "🔧 Processing $rel_file in $project_dir"
        
        # Format & lint
        run_in_project "npx prettier --write '$rel_file'" "$project_dir"
        run_in_project "npx eslint --fix '$rel_file'" "$project_dir"
        
        # Smart actions based on file type
        case "$file" in
            *component*.tsx|*Component*.tsx)
                log "📦 React component detected - checking imports"
                run_in_project "npx tsc --noEmit" "$project_dir"
                ;;
            *hook*.ts|*Hook*.ts|*use*.ts)
                log "🪝 React hook detected - running hook linter"
                run_in_project "npx eslint '$rel_file' --rule 'react-hooks/rules-of-hooks: error'" "$project_dir"
                ;;
            *api*.ts|*API*.ts)
                log "🌐 API file detected - checking types"
                run_in_project "npx tsc --noEmit" "$project_dir"
                ;;
            *types*.ts|*Types*.ts)
                log "📘 Types file detected - validating TypeScript"
                run_in_project "npx tsc --noEmit --strict" "$project_dir"
                ;;
        esac
        
        # Auto-run related tests if they exist
        test_file="${file%.*}.test.${file##*.}"
        if [[ -f "$test_file" ]]; then
            log "🧪 Running related tests"
            run_in_project "npm test -- '$test_file' --watchAll=false" "$project_dir"
        fi
        
        log "✅ Smart processing complete for $rel_file"
    fi
done