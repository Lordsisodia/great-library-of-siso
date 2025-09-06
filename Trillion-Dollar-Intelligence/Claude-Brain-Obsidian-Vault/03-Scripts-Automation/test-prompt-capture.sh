#!/bin/bash

echo "🧪 TESTING PROMPT CAPTURE SYSTEM"
echo "================================"
echo ""
echo "This will show what happens when you submit a prompt..."
echo ""

# Simulate what happens when you type a prompt
TEST_PROMPT="this is my real test prompt from Claude Code"
echo "Simulating prompt: \"$TEST_PROMPT\""
echo ""

# Run through the optimizer
echo "Running optimizer..."
~/.claude/scripts/openai-prompt-optimizer-gpt5-visual.sh "$TEST_PROMPT" > /dev/null 2>&1

# Check what was captured
echo ""
echo "✅ What was captured:"
echo "-------------------"
echo "Original prompt:"
cat ~/.claude/cache/openai-optimizer/last-original.txt
echo ""
echo ""
echo "Log entry:"
tail -1 ~/.claude/logs/openai-gpt5-optimizer.log
echo ""
echo "-------------------"
echo ""
echo "If you see your prompt above, the system is working!"
echo "Your Telegram should have received a notification."