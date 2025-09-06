#!/bin/bash

# 10x Auto Commit Enhancement - Smart git workflow automation
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/hooks.log; }

# Pre-commit quality gates
run_quality_gates() {
    log "🔍 Running pre-commit quality gates"
    
    # Auto-format all staged files
    staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(ts|tsx|js|jsx|md)$' || true)
    
    if [[ -n "$staged_files" ]]; then
        log "📝 Auto-formatting staged files: $staged_files"
        echo "$staged_files" | xargs npx prettier --write 2>/dev/null || true
        echo "$staged_files" | xargs git add 2>/dev/null || true
    fi
    
    # Run lint on TypeScript files
    ts_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(ts|tsx)$' || true)
    if [[ -n "$ts_files" ]] && [[ -f "package.json" ]]; then
        log "🔧 Running TypeScript checks"
        npx tsc --noEmit 2>/dev/null || log "⚠️ TypeScript errors detected"
        echo "$ts_files" | xargs npx eslint --fix 2>/dev/null || true
    fi
}

# Smart commit message enhancement
enhance_commit_message() {
    local original_msg="$1"
    
    # Detect commit type from staged files
    local commit_type=""
    local scope=""
    
    # Analyze staged files for smart commit type detection
    if git diff --cached --name-only | grep -q "test\|spec"; then
        commit_type="test"
    elif git diff --cached --name-only | grep -q "package\.json"; then
        commit_type="deps"
    elif git diff --cached --name-only | grep -q "README\|\.md"; then
        commit_type="docs"
    elif git diff --cached --name-only | grep -q "\.tsx\|\.ts"; then
        if git diff --cached | grep -q "^+.*function\|^+.*const.*="; then
            commit_type="feat"
        else
            commit_type="fix"
        fi
    fi
    
    # Detect SISO project scope
    if pwd | grep -q "SISO-CLIENT-BASE"; then
        scope="client"
    elif pwd | grep -q "SISO-CORE"; then
        scope="core"
    elif pwd | grep -q "SISO-DEV-TOOLS"; then
        scope="tools"
    fi
    
    # Auto-enhance if message doesn't follow conventional format
    if [[ ! "$original_msg" =~ ^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: ]]; then
        local enhanced_msg="${commit_type}"
        [[ -n "$scope" ]] && enhanced_msg="${enhanced_msg}(${scope})"
        enhanced_msg="${enhanced_msg}: ${original_msg}"
        log "🎯 Enhanced commit message: $enhanced_msg"
        echo "$enhanced_msg"
    else
        echo "$original_msg"
    fi
}

# Auto-add Claude co-authorship
add_claude_coauthor() {
    local commit_msg="$1"
    
    if [[ ! "$commit_msg" == *"Co-authored-by: Claude"* ]]; then
        echo -e "$commit_msg\n\n🤖 Generated with Claude Code\n\nCo-authored-by: Claude <noreply@anthropic.com>"
    else
        echo "$commit_msg"
    fi
}

# Smart branch checking
check_branch_safety() {
    local current_branch=$(git branch --show-current)
    
    if [[ "$current_branch" == "main" ]] || [[ "$current_branch" == "master" ]]; then
        log "⚠️ WARNING: Committing directly to $current_branch"
        return 1
    fi
    
    log "✅ Safe to commit on branch: $current_branch"
    return 0
}

# Main execution
log "🚀 Auto-commit enhancement started"

# Run quality gates
run_quality_gates

# Check branch safety
if ! check_branch_safety; then
    log "❌ Unsafe branch detected - commit blocked"
    exit 1
fi

log "✅ Auto-commit enhancement complete"