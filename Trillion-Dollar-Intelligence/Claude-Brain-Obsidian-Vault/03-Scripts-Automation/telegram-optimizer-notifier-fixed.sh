#!/bin/bash

# 🤖 Fixed Telegram Notifier - Shows REAL prompts and metrics
# This version properly captures and displays all prompt data

# CONFIGURATION
TELEGRAM_BOT_TOKEN="8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w"
TELEGRAM_CHAT_ID="7643203581"

# Paths
CACHE_DIR="$HOME/.claude/cache/openai-optimizer"
LOG_DIR="$HOME/.claude/logs"

# Function to send Telegram message
send_telegram() {
    local message="$1"
    
    # Send message
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=HTML" \
        -d "disable_web_page_preview=true" > /dev/null 2>&1
}

echo "🤖 Fixed Telegram Optimizer Notifier Started!"
echo "📱 Sending to chat ID: $TELEGRAM_CHAT_ID"
echo "Monitoring for prompt optimizations..."
echo ""

# Send startup message
send_telegram "🚀 <b>Claude Optimizer Monitor Started!</b>

I'm now watching for prompt optimizations.
You'll receive notifications with:
• Your original prompts
• Optimized versions  
• Real expansion metrics
• Which agent processed them"

# Monitor logs continuously
tail -f "$LOG_DIR/openai-gpt5-optimizer.log" 2>/dev/null | while read -r line; do
    
    # Detect optimization completion
    if [[ "$line" =~ "optimization complete:" ]] && [[ "$line" =~ ([0-9]+)[[:space:]]→[[:space:]]([0-9]+)[[:space:]]chars[[:space:]]\(([0-9]+)%[[:space:]]expansion\) ]]; then
        
        # Extract metrics from the log line
        ORIG_CHARS="${BASH_REMATCH[1]}"
        OPT_CHARS="${BASH_REMATCH[2]}"
        EXPANSION="${BASH_REMATCH[3]}"
        
        # Get timestamp
        if [[ "$line" =~ \[([0-9]{4}-[0-9]{2}-[0-9]{2}[[:space:]][0-9]{2}:[0-9]{2}:[0-9]{2})\] ]]; then
            TIMESTAMP="${BASH_REMATCH[1]}"
        else
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        fi
        
        # Get the actual prompts
        ORIGINAL_PROMPT="[Not captured]"
        OPTIMIZED_PREVIEW="[Not captured]"
        
        if [[ -f "$CACHE_DIR/last-original.txt" ]]; then
            ORIGINAL_PROMPT=$(cat "$CACHE_DIR/last-original.txt" | tr '\n' ' ' | head -c 200)
            if [[ ${#ORIGINAL_PROMPT} -ge 200 ]]; then
                ORIGINAL_PROMPT="${ORIGINAL_PROMPT}..."
            fi
        fi
        
        if [[ -f "$CACHE_DIR/last-optimized.txt" ]]; then
            # Get first few lines of optimized prompt
            OPTIMIZED_PREVIEW=$(head -n 5 "$CACHE_DIR/last-optimized.txt" | grep -E "^(#|You are|Your|The)" | head -1)
            if [[ -z "$OPTIMIZED_PREVIEW" ]]; then
                OPTIMIZED_PREVIEW=$(head -n 1 "$CACHE_DIR/last-optimized.txt")
            fi
        fi
        
        # Calculate expansion factor
        EXPANSION_FACTOR=$(echo "scale=1; $OPT_CHARS / $ORIG_CHARS" | bc 2>/dev/null || echo "N/A")
        
        # Create message
        MESSAGE="🧠 <b>Prompt Optimized!</b>

<b>Agent:</b> GPT-5 Enhanced Optimizer
<b>Time:</b> ${TIMESTAMP}

📊 <b>Real Metrics:</b>
• Original: ${ORIG_CHARS} chars
• Optimized: ${OPT_CHARS} chars
• Expansion: ${EXPANSION}% (${EXPANSION_FACTOR}x)

📝 <b>Your Original Prompt:</b>
<code>${ORIGINAL_PROMPT}</code>

✨ <b>Optimized To:</b>
<i>${OPTIMIZED_PREVIEW}</i>

#ClaudeOptimizer #GPT5"
        
        # Send notification
        send_telegram "$MESSAGE"
        
        echo "[$TIMESTAMP] Sent notification - Original: $ORIG_CHARS chars → Optimized: $OPT_CHARS chars"
    fi
    
    # Also detect when optimization starts
    if [[ "$line" =~ "Starting GPT-5 enhanced optimization" ]]; then
        send_telegram "⚡ <b>Optimization Starting...</b>
Processing your prompt with GPT-5 framework!"
    fi
done