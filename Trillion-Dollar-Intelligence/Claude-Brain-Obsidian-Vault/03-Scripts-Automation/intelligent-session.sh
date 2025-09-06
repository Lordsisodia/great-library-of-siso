#!/bin/bash

# 10x Intelligent Session Manager - Learn and adapt from prompts
PROMPT="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DATE=$(date '+%Y-%m-%d')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/hooks.log; }

# Smart prompt analysis and auto-suggestions
analyze_prompt() {
    local prompt="$1"
    
    # Detect common patterns and provide intelligent automation
    case "$prompt" in
        *"fix"*|*"bug"*|*"error"*)
            log "🐛 Bug fix session detected - enabling debug mode"
            echo "DEBUG_MODE=true" > ~/.claude/session-context
            ;;
        *"feature"*|*"add"*|*"create"*)
            log "✨ Feature development detected - enabling productivity mode"
            echo "FEATURE_MODE=true" > ~/.claude/session-context
            ;;
        *"refactor"*|*"optimize"*|*"improve"*)
            log "🔧 Refactoring session detected - enabling quality mode"
            echo "REFACTOR_MODE=true" > ~/.claude/session-context
            ;;
        *"test"*|*"testing"*|*"spec"*)
            log "🧪 Testing session detected - enabling test mode"
            echo "TEST_MODE=true" > ~/.claude/session-context
            ;;
        *"deploy"*|*"production"*|*"release"*)
            log "🚀 Deployment session detected - enabling safety mode"
            echo "DEPLOY_MODE=true" > ~/.claude/session-context
            ;;
    esac
}

# Auto-suggest next actions based on session patterns
suggest_actions() {
    local prompt="$1"
    
    if [[ "$prompt" == *"SISO"* ]]; then
        log "🏢 SISO project session - suggesting quality checks"
        # Auto-suggest running lint and build for SISO projects
        if command -v npm >/dev/null && [[ -f "package.json" ]]; then
            log "💡 Auto-suggestion: Run 'npm run lint && npm run build' for quality assurance"
        fi
    fi
    
    if [[ "$prompt" == *"component"* ]] && [[ "$prompt" == *"React"* ]]; then
        log "⚛️ React component work detected - suggesting Storybook check"
    fi
}

# Smart session tracking with insights
track_session() {
    local prompt="$1"
    local sanitized=$(echo "$prompt" | sed 's/[A-Za-z0-9._%+-]*@[A-Za-z0-9.-]*\.[A-Za-z]\{2,\}/[EMAIL]/g')
    
    # Session analytics
    echo "[$TIMESTAMP] SESSION: $sanitized" >> ~/.claude/logs/session-$DATE.log
    echo "$TIMESTAMP,${#prompt},$(echo "$prompt" | wc -w)" >> ~/.claude/analytics/daily-metrics.csv
    
    # Keep rolling 7-day window
    find ~/.claude/logs -name "session-*.log" -mtime +7 -delete 2>/dev/null || true
}

# Initialize directories
mkdir -p ~/.claude/analytics

# Run analysis
analyze_prompt "$PROMPT"
suggest_actions "$PROMPT"
track_session "$PROMPT"

log "🧠 Intelligent session processing complete"