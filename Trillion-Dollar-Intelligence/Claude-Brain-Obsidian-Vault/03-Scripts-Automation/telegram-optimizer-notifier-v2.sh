#!/bin/bash

# 🤖 Telegram Notification Agent V2 - Shows REAL prompts and accurate metrics
# Sends detailed notifications whenever ANY agent optimizes a prompt

# CONFIGURATION
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-7643203581}"

# Paths
CACHE_DIR="$HOME/.claude/cache/openai-optimizer"
LOG_DIR="$HOME/.claude/logs"
VISUAL_LOG="$HOME/.claude/visual-optimizer-output.log"

# Function to send Telegram message
send_telegram() {
    local message="$1"
    
    if [[ "$TELEGRAM_CHAT_ID" == "YOUR_CHAT_ID_HERE" ]]; then
        echo "⚠️ Please set TELEGRAM_CHAT_ID! Run: ~/.claude/scripts/get-telegram-chat-id.sh"
        return 1
    fi
    
    # URL encode the message
    local encoded_message=$(echo -n "$message" | jq -sRr @uri)
    
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=Markdown" \
        -d "disable_web_page_preview=true" > /dev/null
}

# Function to get real metrics from logs
get_real_metrics() {
    local log_file="$1"
    local timestamp="$2"
    
    # Look for the exact optimization line
    local metrics=$(grep -A1 -B1 "$timestamp" "$log_file" | grep "optimization complete:" | tail -1)
    
    if [[ "$metrics" =~ ([0-9]+)[[:space:]]→[[:space:]]([0-9]+)[[:space:]]chars[[:space:]]\(([0-9]+)%[[:space:]]expansion\) ]]; then
        echo "${BASH_REMATCH[1]}|${BASH_REMATCH[2]}|${BASH_REMATCH[3]}"
    else
        echo "0|0|0"
    fi
}

# Function to get prompts from cache
get_prompt_data() {
    local original=""
    local optimized=""
    
    # Get original prompt
    if [[ -f "$CACHE_DIR/last-original.txt" ]]; then
        original=$(cat "$CACHE_DIR/last-original.txt" | head -c 200)
        if [[ ${#original} -eq 200 ]]; then
            original="${original}..."
        fi
    fi
    
    # Get optimized prompt preview
    if [[ -f "$CACHE_DIR/last-optimized.txt" ]]; then
        # Get first meaningful lines after headers
        optimized=$(grep -E "^(- |You are|Your |The )" "$CACHE_DIR/last-optimized.txt" | head -3 | tr '\n' ' ' | head -c 150)
        if [[ -z "$optimized" ]]; then
            optimized=$(head -c 150 "$CACHE_DIR/last-optimized.txt")
        fi
        optimized="${optimized}..."
    fi
    
    echo "$original|$optimized"
}

echo "🤖 Telegram Optimizer Notifier V2 Started!"
echo "📊 Showing REAL prompts and accurate metrics"
echo "Monitoring for prompt optimizations..."
echo ""

# Monitor the specific GPT-5 optimizer log
tail -f "$LOG_DIR/openai-gpt5-optimizer.log" "$LOG_DIR/openai-optimizer.log" 2>/dev/null | while read -r line; do
    
    # Check for optimization completion
    if [[ "$line" =~ "optimization complete:" ]]; then
        
        # Extract timestamp from log line
        if [[ "$line" =~ \[([0-9]{4}-[0-9]{2}-[0-9]{2}[[:space:]][0-9]{2}:[0-9]{2}:[0-9]{2})\] ]]; then
            TIMESTAMP="${BASH_REMATCH[1]}"
        else
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        fi
        
        # Determine which agent
        AGENT_NAME="Unknown"
        LOG_FILE=""
        if [[ "$line" =~ "gpt5" ]]; then
            AGENT_NAME="GPT-5 Enhanced Optimizer"
            LOG_FILE="$LOG_DIR/openai-gpt5-optimizer.log"
        elif [[ "$line" =~ "openai" ]]; then
            AGENT_NAME="Standard OpenAI Optimizer"
            LOG_FILE="$LOG_DIR/openai-optimizer.log"
        fi
        
        # Get real metrics
        IFS='|' read -r orig_chars opt_chars expansion <<< "$(get_real_metrics "$LOG_FILE" "$TIMESTAMP")"
        
        # Get actual prompts
        IFS='|' read -r original_prompt optimized_preview <<< "$(get_prompt_data)"
        
        # Format numbers nicely
        orig_formatted=$(printf "%'d" "$orig_chars" 2>/dev/null || echo "$orig_chars")
        opt_formatted=$(printf "%'d" "$opt_chars" 2>/dev/null || echo "$opt_chars")
        
        # Create detailed message
        MESSAGE="🧠 *Prompt Optimization Complete!*

*Agent:* ${AGENT_NAME}
*Time:* ${TIMESTAMP}

📊 *Real Metrics:*
• Original: ${orig_formatted} chars
• Optimized: ${opt_formatted} chars  
• Expansion: ${expansion}% ($(echo "scale=1; $opt_chars/$orig_chars" | bc -l 2>/dev/null || echo "N/A")x)

📝 *Original Prompt:*
\`${original_prompt}\`

✨ *Optimized Preview:*
_${optimized_preview}_

#ClaudeOptimizer #RealMetrics"
        
        # Send notification
        send_telegram "$MESSAGE"
        
        # Log locally
        echo "[$TIMESTAMP] Sent detailed notification for $AGENT_NAME"
        echo "  Original: $orig_chars chars → Optimized: $opt_chars chars (${expansion}% expansion)"
    fi
    
    # Also monitor for visual optimizer activation
    if [[ "$line" =~ "Starting GPT-5 enhanced optimization" ]]; then
        # Quick notification for start
        send_telegram "⚡ *Optimization Started*
GPT-5 Optimizer is processing a prompt...
Results coming soon!"
    fi
done