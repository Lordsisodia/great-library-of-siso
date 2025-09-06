#!/bin/bash

# 🤖 Telegram Notification Agent for Prompt Optimization
# Sends notifications whenever ANY agent optimizes a prompt

# CONFIGURATION - You need to set these!
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-YOUR_CHAT_ID_HERE}"

# Function to send Telegram message
send_telegram() {
    local message="$1"
    local parse_mode="${2:-Markdown}"
    
    if [[ "$TELEGRAM_BOT_TOKEN" == "YOUR_BOT_TOKEN_HERE" ]]; then
        echo "⚠️ Please set TELEGRAM_BOT_TOKEN in this script or as environment variable"
        return 1
    fi
    
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${message}" \
        -d "parse_mode=${parse_mode}" \
        -d "disable_web_page_preview=true" > /dev/null
}

# Function to extract prompt info
get_prompt_info() {
    local log_line="$1"
    local agent_name="$2"
    
    # Extract expansion percentage if available
    if [[ "$log_line" =~ ([0-9]+)%[[:space:]]+expansion ]]; then
        echo "${BASH_REMATCH[1]}%"
    else
        echo "N/A"
    fi
}

# Agent detection patterns
declare -A AGENT_PATTERNS=(
    ["GPT-5 Optimizer"]="openai-gpt5-optimizer"
    ["Standard OpenAI"]="openai-prompt-optimizer"
    ["Prompt Enhancer"]="prompt-enhancer"
    ["Security Validator"]="security-validator"
    ["Intelligent Session"]="intelligent-session"
)

# Monitor all optimizer logs
echo "🤖 Telegram Optimizer Notifier Started!"
echo "Monitoring for prompt optimizations..."
echo ""

# Create notification throttle directory
mkdir -p ~/.claude/telegram-notifications

# Monitor multiple log files
tail -f ~/.claude/logs/*.log 2>/dev/null | while read -r line; do
    # Check if this is an optimization completion
    if [[ "$line" =~ "optimization complete:" ]] || [[ "$line" =~ "Enhanced analysis saved" ]]; then
        
        # Detect which agent based on log content
        AGENT_NAME="Unknown Agent"
        for agent in "${!AGENT_PATTERNS[@]}"; do
            pattern="${AGENT_PATTERNS[$agent]}"
            if [[ "$line" =~ $pattern ]]; then
                AGENT_NAME="$agent"
                break
            fi
        done
        
        # Extract metrics
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        EXPANSION=$(get_prompt_info "$line" "$AGENT_NAME")
        
        # Get original prompt from cache if available
        ORIGINAL_PROMPT="N/A"
        if [[ -f ~/.claude/cache/openai-optimizer/last-original.txt ]]; then
            ORIGINAL_PROMPT=$(head -c 50 ~/.claude/cache/openai-optimizer/last-original.txt)
            if [[ ${#ORIGINAL_PROMPT} -eq 50 ]]; then
                ORIGINAL_PROMPT="${ORIGINAL_PROMPT}..."
            fi
        fi
        
        # Compose Telegram message
        MESSAGE="🧠 *Prompt Optimized!*

*Agent:* ${AGENT_NAME}
*Time:* ${TIMESTAMP}
*Expansion:* ${EXPANSION}
*Original:* _${ORIGINAL_PROMPT}_

#ClaudeOptimizer #${AGENT_NAME// /}"
        
        # Send notification
        send_telegram "$MESSAGE"
        
        # Log locally
        echo "[$TIMESTAMP] Sent notification for $AGENT_NAME optimization"
        
        # Save to history
        echo "[$TIMESTAMP] Agent: $AGENT_NAME, Expansion: $EXPANSION" >> ~/.claude/telegram-notifications/history.log
    fi
    
    # Also detect visual optimizer output
    if [[ "$line" =~ "OPENAI GPT-5 PROMPT OPTIMIZER ACTIVATED" ]]; then
        MESSAGE="🚀 *Visual Optimizer Activated!*

The GPT-5 Enhanced Optimizer is transforming a prompt with visual feedback.

Check your terminal or log file for details!

#VisualOptimizer #GPT5"
        
        send_telegram "$MESSAGE"
    fi
done