#!/bin/bash

# 10x Smart Dependency Manager - Auto-optimize package.json changes
FILE_PATH="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/hooks.log; }

log "📦 Smart dependency management for $FILE_PATH"

# Auto-install new dependencies
if [[ -f "$FILE_PATH" ]]; then
    project_dir=$(dirname "$FILE_PATH")
    cd "$project_dir"
    
    # Check if dependencies changed
    if git diff HEAD~ -- package.json | grep -q "^\+.*dependencies\|^\+.*devDependencies"; then
        log "🔄 New dependencies detected - auto-installing"
        npm install --silent 2>/dev/null || log "⚠️ Install failed"
    fi
    
    # Security audit on dependency changes
    if command -v npm >/dev/null; then
        log "🛡️ Running security audit"
        npm audit --audit-level moderate >> ~/.claude/logs/security-audit.log 2>&1 || true
        
        # Auto-fix non-breaking vulnerabilities
        npm audit fix --only=prod --silent 2>/dev/null || true
    fi
    
    # Check for unused dependencies
    if command -v npx >/dev/null; then
        log "🧹 Checking for unused dependencies"
        npx depcheck --json > ~/.claude/tmp/depcheck.json 2>/dev/null || true
        
        if [[ -f ~/.claude/tmp/depcheck.json ]]; then
            unused=$(cat ~/.claude/tmp/depcheck.json | grep -o '"unused":\[[^]]*\]' | grep -o '"[^"]*"' | wc -l)
            if [[ $unused -gt 0 ]]; then
                log "📊 Found $unused potentially unused dependencies"
            fi
        fi
    fi
    
    # Auto-update lock file
    if [[ -f "package-lock.json" ]]; then
        log "🔒 Updating lock file"
        git add package-lock.json 2>/dev/null || true
    fi
    
    # SISO-specific optimizations
    if grep -q "SISO" package.json 2>/dev/null; then
        log "🏢 SISO project detected - applying optimizations"
        
        # Ensure consistent versions across SISO projects
        if grep -q "react.*18" package.json && ! grep -q "react.*18.2" package.json; then
            log "⚡ Suggesting React 18.2+ for SISO consistency"
        fi
    fi
fi

log "✅ Smart dependency management complete"