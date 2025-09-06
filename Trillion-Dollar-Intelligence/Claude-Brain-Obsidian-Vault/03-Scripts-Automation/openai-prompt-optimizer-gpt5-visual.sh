#!/bin/bash

# 🧠 OpenAI Framework + GPT-5 Enhanced Prompt Optimizer WITH VISUAL FEEDBACK
# Shows the transformation happening in real-time!

ORIGINAL_PROMPT="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
PROJECT_ROOT=$(pwd)

# Logging setup
LOG_FILE="$HOME/.claude/logs/openai-gpt5-optimizer.log"
CACHE_DIR="$HOME/.claude/cache/openai-optimizer"
mkdir -p "$(dirname "$LOG_FILE")" "$CACHE_DIR"

log() { echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"; }

# Visual feedback function
show_transformation() {
    local original="$1"
    local optimized="$2"
    local original_length=${#original}
    local optimized_length=${#optimized}
    local expansion=$(( (optimized_length * 100) / original_length ))
    
    # Save visual output to log file
    VISUAL_LOG="$HOME/.claude/visual-optimizer-output.log"
    
    {
        echo ""
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
        echo ""
        echo "🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀"
        echo "🧠 OPENAI GPT-5 PROMPT OPTIMIZER ACTIVATED! 🧠"
        echo "🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀"
    } > "$VISUAL_LOG"
    
    # Also send notification
    ~/.claude/scripts/cross-platform-notifier.sh "🧠 Prompt Optimized: ${expansion}% expansion!" "GPT-5 Optimizer" "low" &
    
    # Print visual separator (this goes to stderr so it might show in console)
    echo "" >&2
    echo "🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀" >&2
    echo "🧠 OPENAI GPT-5 PROMPT OPTIMIZER ACTIVATED! 🧠" >&2
    echo "🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀" >&2
    {
        echo ""
        echo "📝 YOUR ORIGINAL PROMPT ($original_length chars):"
        echo "┌────────────────────────────────────────┐"
        echo "│ $original"
        echo "└────────────────────────────────────────┘"
        echo ""
        echo "⚡ TRANSFORMATION IN PROGRESS..."
        echo ""
        echo "✨ OPTIMIZED PROMPT ($optimized_length chars - ${expansion}% expansion):"
        echo "┌────────────────────────────────────────┐"
        echo "$optimized" | head -15 | sed 's/^/│ /'
        echo "│ ... [FULL OPTIMIZATION APPLIED]"
        echo "└────────────────────────────────────────┘"
        echo ""
        echo "🎯 ENHANCEMENTS APPLIED:"
        echo "  ✅ Expert Role Definition"
        echo "  ✅ Agentic Persistence Protocol"
        echo "  ✅ Tool Communication Preambles"
        echo "  ✅ Smart Context Gathering"
        echo "  ✅ Excellence Framework"
        echo "  ✅ Structured Output Format"
        echo ""
        echo "🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀"
        echo ""
        echo "📄 Full visual output saved to: ~/.claude/visual-optimizer-output.log"
        echo ""
    } >> "$VISUAL_LOG"
}

# Import all the optimization functions from the original script
source ~/.claude/scripts/openai-prompt-optimizer-gpt5.sh >/dev/null 2>&1

# Main optimization with visual feedback
optimize_prompt_visual() {
    local original="$1"
    local optimized="$original"
    
    log "🧠 Starting GPT-5 enhanced optimization with visual feedback..."
    
    # Skip optimization for very short or already well-structured prompts
    if [[ ${#original} -lt 20 ]] || [[ "$original" =~ ^#.*Role.*Objective ]]; then
        log "ℹ️ Prompt already optimized or too short, skipping"
        echo "$original"
        return
    fi
    
    # Apply GPT-5 enhanced framework rules in sequence
    optimized=$(add_enhanced_role "$optimized")
    optimized=$(add_agentic_persistence "$optimized")
    optimized=$(add_tool_preambles "$optimized")
    optimized=$(add_smart_context_gathering "$optimized")
    optimized=$(add_self_reflection_framework "$optimized")
    optimized=$(add_enhanced_instructions "$optimized")
    optimized=$(add_enhanced_chain_of_thought "$optimized")
    optimized=$(add_enhanced_project_context "$optimized")
    optimized=$(add_enhanced_output_format "$optimized")
    
    # Add final GPT-5 completion instructions
    optimized="$optimized
---
**Final Instructions**: Work autonomously and systematically. Use available tools to gather information rather than making assumptions. Only complete your turn when you're certain the task is fully resolved.

*🧠 Enhanced with GPT-5 Framework | OpenAI Best Practices + Agentic Intelligence*"
    
    # Show the transformation visually
    show_transformation "$original" "$optimized"
    
    # Save optimization results
    echo "$original" > "$CACHE_DIR/last-original.txt"
    echo "$optimized" > "$CACHE_DIR/last-optimized.txt"
    
    # Calculate improvement metrics
    local original_length=${#original}
    local optimized_length=${#optimized}
    local improvement_ratio=$(( (optimized_length * 100) / original_length ))
    
    log "✅ GPT-5 optimization complete: ${original_length} → ${optimized_length} chars (${improvement_ratio}% expansion)"
    
    # Return the optimized prompt for Claude to use
    echo "$optimized"
}

# Analyze prompt first
analysis_file=$(analyze_prompt_structure "$ORIGINAL_PROMPT")
log "📊 Enhanced analysis saved to: $analysis_file"

# Apply GPT-5 enhanced optimization with visual feedback
optimized_result=$(optimize_prompt_visual "$ORIGINAL_PROMPT")

# Output the optimized prompt
echo "$optimized_result"