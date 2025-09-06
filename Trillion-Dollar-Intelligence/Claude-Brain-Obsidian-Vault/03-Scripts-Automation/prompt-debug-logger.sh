#!/bin/bash

# Debug logger to see EVERY prompt that comes through
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
PROMPT="$1"

# Log to a debug file
{
    echo "[$TIMESTAMP] PROMPT RECEIVED:"
    echo "Length: ${#PROMPT} chars"
    echo "Content: $PROMPT"
    echo "---"
} >> ~/.claude/logs/prompt-debug.log

# Also send to Telegram immediately
TELEGRAM_BOT_TOKEN="8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w"
TELEGRAM_CHAT_ID="7643203581"

MESSAGE="🔍 <b>Debug: Prompt Received</b>
Time: $TIMESTAMP
Length: ${#PROMPT} chars
Prompt: <code>$PROMPT</code>"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "text=${MESSAGE}" \
    -d "parse_mode=HTML" > /dev/null 2>&1

# Pass through to optimizer
~/.claude/scripts/openai-prompt-optimizer-gpt5-visual.sh "$PROMPT"