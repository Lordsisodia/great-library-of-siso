#!/bin/bash

# 🧠 OpenAI Optimizer Activation Script
# Switches between normal and OpenAI-optimized prompt hooks

CLAUDE_DIR="$HOME/.claude"
CURRENT_HOOKS="$CLAUDE_DIR/settings.hooks.json"
OPENAI_HOOKS="$CLAUDE_DIR/settings.hooks.openai-enhanced.json"
BACKUP_HOOKS="$CLAUDE_DIR/settings.hooks.json.backup"

case "$1" in
    "on"|"enable"|"activate")
        MODE="${2:-standard}"
        
        if [[ "$MODE" == "gpt5" ]]; then
            echo "🧠 Activating GPT-5 Enhanced OpenAI Prompt Optimizer..."
            
            # Create GPT-5 enhanced hooks configuration
            sed 's/openai-prompt-optimizer\.sh/openai-prompt-optimizer-gpt5.sh/g' "$OPENAI_HOOKS" > "$CLAUDE_DIR/settings.hooks.gpt5-enhanced.json"
            
            # Backup current hooks if not already backed up
            if [[ ! -f "$BACKUP_HOOKS" ]]; then
                cp "$CURRENT_HOOKS" "$BACKUP_HOOKS"
                echo "✅ Current hooks backed up to: settings.hooks.json.backup"
            fi
            
            # Switch to GPT-5 enhanced hooks
            cp "$CLAUDE_DIR/settings.hooks.gpt5-enhanced.json" "$CURRENT_HOOKS"
            echo "🚀 GPT-5 Enhanced Optimizer ACTIVATED!"
            echo "🧠 Your prompts will now be optimized using GPT-5 framework + agentic intelligence"
        else
            echo "🧠 Activating Standard OpenAI Prompt Optimizer..."
            
            # Backup current hooks if not already backed up
            if [[ ! -f "$BACKUP_HOOKS" ]]; then
                cp "$CURRENT_HOOKS" "$BACKUP_HOOKS"
                echo "✅ Current hooks backed up to: settings.hooks.json.backup"
            fi
            
            # Switch to OpenAI-enhanced hooks
            cp "$OPENAI_HOOKS" "$CURRENT_HOOKS"
            echo "🚀 OpenAI Prompt Optimizer ACTIVATED!"
            echo "📝 Your prompts will now be optimized using OpenAI's framework"
        fi
        ;;
        
    "off"|"disable"|"deactivate")
        echo "🔄 Deactivating OpenAI Prompt Optimizer..."
        
        if [[ -f "$BACKUP_HOOKS" ]]; then
            cp "$BACKUP_HOOKS" "$CURRENT_HOOKS"
            echo "✅ Restored original hooks configuration"
        else
            echo "⚠️ No backup found, keeping current configuration"
        fi
        ;;
        
    "status")
        if grep -q "openai-prompt-optimizer-gpt5.sh" "$CURRENT_HOOKS" 2>/dev/null; then
            echo "🧠 GPT-5 Enhanced Optimizer: ACTIVE"
            echo "📊 Check logs: ~/.claude/logs/openai-gpt5-optimizer.log"
            echo "💾 Cache location: ~/.claude/cache/openai-optimizer/"
        elif grep -q "openai-prompt-optimizer.sh" "$CURRENT_HOOKS" 2>/dev/null; then
            echo "🧠 Standard OpenAI Optimizer: ACTIVE"
            echo "📊 Check logs: ~/.claude/logs/openai-optimizer.log"
            echo "💾 Cache location: ~/.claude/cache/openai-optimizer/"
        else
            echo "😴 OpenAI Prompt Optimizer: INACTIVE"
        fi
        ;;
        
    "test")
        echo "🧪 Testing OpenAI Prompt Optimizer..."
        
        # Create test prompts
        test_prompts=(
            "fix this bug"
            "create a login component"
            "help me debug the API issue"
            "implement user authentication"
        )
        
        for prompt in "${test_prompts[@]}"; do
            echo ""
            echo "Testing prompt: '$prompt'"
            echo "----------------------------------------"
            ~/.claude/scripts/openai-prompt-optimizer.sh "$prompt" | head -10
            echo "----------------------------------------"
        done
        ;;
        
    *)
        echo "🧠 OpenAI Prompt Optimizer Control"
        echo ""
        echo "Usage: $0 {on [mode]|off|status|test}"
        echo ""
        echo "Commands:"
        echo "  on           - Activate standard OpenAI prompt optimization"
        echo "  on gpt5      - Activate GPT-5 enhanced optimization (recommended)"
        echo "  off          - Deactivate and restore original hooks"
        echo "  status       - Check current optimizer status"
        echo "  test         - Test the optimizer with sample prompts"
        echo ""
        echo "Modes:"
        echo "  standard     - OpenAI framework (GPT-4.1 patterns)"
        echo "  gpt5         - Enhanced with agentic intelligence, tool preambles,"
        echo "                 persistence, and advanced context gathering"
        echo ""
        echo "Current status:"
        $0 status
        ;;
esac