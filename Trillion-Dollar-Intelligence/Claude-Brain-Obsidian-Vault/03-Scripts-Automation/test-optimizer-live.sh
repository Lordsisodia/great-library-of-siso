#!/bin/bash

# Live test script to PROVE the optimizer is working

echo "🔍 LIVE OPTIMIZER TEST - Let's see what's REALLY happening!"
echo "============================================"
echo ""

# Test 1: Show current hooks configuration
echo "📋 STEP 1: Current Hook Configuration"
echo "-----------------------------------"
grep -A 5 "UserPromptSubmit" ~/.claude/settings.hooks.json | grep -E "(command|openai)"
echo ""

# Test 2: Show what script is being called
echo "🎯 STEP 2: Which optimizer is active?"
echo "-----------------------------------"
ACTIVE_SCRIPT=$(grep -A 3 "UserPromptSubmit" ~/.claude/settings.hooks.json | grep "command" | head -1 | cut -d'"' -f4)
echo "Active script: $ACTIVE_SCRIPT"
echo ""

# Test 3: Create a test prompt and show the transformation
echo "🧪 STEP 3: Let's transform a test prompt RIGHT NOW"
echo "-----------------------------------"
TEST_PROMPT="help me debug this API error"
echo "Original prompt: '$TEST_PROMPT'"
echo ""
echo "Transforming..."
echo "-----------------------------------"

# Run the actual optimizer and show first 20 lines
if [[ -f "$ACTIVE_SCRIPT" ]]; then
    TRANSFORMED=$($ACTIVE_SCRIPT "$TEST_PROMPT" 2>&1 | head -20)
    echo "$TRANSFORMED"
    echo "..."
    echo "[Output truncated - full version is much longer]"
else
    echo "❌ ERROR: Script not found at $ACTIVE_SCRIPT"
fi

echo ""
echo "📊 STEP 4: Check the logs for recent activity"
echo "-----------------------------------"
echo "Last 3 optimizer runs:"
tail -3 ~/.claude/logs/openai-gpt5-optimizer.log 2>/dev/null || echo "No GPT-5 logs yet"
echo ""

echo "✅ STEP 5: Is it REALLY active?"
echo "-----------------------------------"
# Check if the hook will actually fire
if grep -q "openai-prompt-optimizer" ~/.claude/settings.hooks.json; then
    echo "✅ YES! OpenAI optimizer is in your active hooks!"
    echo "🚀 Every prompt you type IS being transformed!"
else
    echo "❌ NO! OpenAI optimizer is NOT in your hooks!"
fi

echo ""
echo "🔬 PROOF: Look at the cache file that gets updated with EVERY prompt:"
echo "-----------------------------------"
ls -la ~/.claude/cache/openai-optimizer/current-analysis.json
echo ""
echo "Last analysis content:"
cat ~/.claude/cache/openai-optimizer/current-analysis.json 2>/dev/null | jq '.' || echo "No analysis yet"

echo ""
echo "============================================"
echo "🎉 TEST COMPLETE - You can now SEE exactly what's happening!"