#!/bin/bash

# SISO Ecosystem - Session Logging Hook
# Logs all Claude prompts for analytics and improvement

PROMPT="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DATE=$(date '+%Y-%m-%d')

# Ensure logs directory exists
mkdir -p ~/.claude/logs

# Log to daily session file
echo "[$TIMESTAMP] USER PROMPT: $PROMPT" >> ~/.claude/logs/session-$DATE.log

# Log to master hooks log
echo "[$TIMESTAMP] Session logged: ${#PROMPT} characters" >> ~/.claude/logs/hooks.log

# Optional: Log to SISO ecosystem analytics (if configured)
if [[ -n "$SISO_ANALYTICS_ENDPOINT" ]]; then
    curl -s -X POST "$SISO_ANALYTICS_ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "{\"timestamp\":\"$TIMESTAMP\",\"prompt_length\":${#PROMPT},\"session_type\":\"claude_code\"}" \
        2>/dev/null || true
fi

exit 0