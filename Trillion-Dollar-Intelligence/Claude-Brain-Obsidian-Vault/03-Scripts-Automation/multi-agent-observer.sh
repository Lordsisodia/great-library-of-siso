#!/bin/bash

# Multi-Agent Observer - Inspired by williamkapke/MadameClaude
TOOL_NAME="$1"
FILE_PATHS="$2"
SESSION_ID="${CLAUDE_SESSION_ID:-$(date +%s)}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Event structure for observability
create_event() {
    local tool="$1"
    local files="$2"
    local event_type="$3"
    
    cat > ~/.claude/events/event-$(date +%s%N).json << EOF
{
  "timestamp": "$TIMESTAMP",
  "sessionId": "$SESSION_ID",
  "toolName": "$tool",
  "filePaths": "$files",
  "eventType": "$event_type",
  "projectPath": "$(pwd)",
  "gitBranch": "$(git branch --show-current 2>/dev/null || echo 'no-git')",
  "userAgent": "SISO-Hooks-v2"
}
EOF
}

# Stream to monitoring system (if available)
stream_event() {
    local event_file="$1"
    
    # Check if monitoring server is running
    if command -v curl >/dev/null && curl -s -f "http://localhost:8080/health" >/dev/null 2>&1; then
        curl -s -X POST "http://localhost:8080/events" \
             -H "Content-Type: application/json" \
             -d @"$event_file" >/dev/null 2>&1 || true
    fi
}

# Color-coded logging (community pattern)
log_with_color() {
    local message="$1"
    local color="$2"
    
    case "$color" in
        "red") echo -e "\033[31m$message\033[0m" ;;
        "green") echo -e "\033[32m$message\033[0m" ;;
        "yellow") echo -e "\033[33m$message\033[0m" ;;
        "blue") echo -e "\033[34m$message\033[0m" ;;
        *) echo "$message" ;;
    esac
    
    echo "$message" >> ~/.claude/logs/observer.log
}

# Agent activity analysis
analyze_activity() {
    local tool="$1"
    
    case "$tool" in
        "Edit"|"Write"|"MultiEdit")
            log_with_color "📝 Code modification event - $tool" "green"
            ;;
        "Bash")
            log_with_color "💻 System command execution" "yellow"
            ;;
        "mcp__*")
            log_with_color "🔌 MCP integration call" "blue"
            ;;
        *)
            log_with_color "🔧 Tool usage: $tool" "white"
            ;;
    esac
}

# Session tracking
track_session() {
    echo "$TIMESTAMP,$SESSION_ID,$TOOL_NAME,$FILE_PATHS" >> ~/.claude/analytics/session-tracking.csv
    
    # Session metrics
    local session_file="~/.claude/sessions/$SESSION_ID.json"
    if [[ ! -f "$session_file" ]]; then
        echo '{"startTime":"'$TIMESTAMP'","toolCount":0,"events":[]}' > "$session_file"
    fi
}

# Initialize directories
mkdir -p ~/.claude/events ~/.claude/sessions ~/.claude/analytics

# Create and stream event
event_file="~/.claude/events/event-$(date +%s%N).json"
create_event "$TOOL_NAME" "$FILE_PATHS" "PostToolUse"
stream_event "$event_file"

# Analysis and tracking
analyze_activity "$TOOL_NAME"
track_session

log_with_color "🔍 Multi-agent observation complete for session $SESSION_ID" "green"