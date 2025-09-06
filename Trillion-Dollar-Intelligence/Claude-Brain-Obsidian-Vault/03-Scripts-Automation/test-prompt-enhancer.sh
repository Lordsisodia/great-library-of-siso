#!/bin/bash

# Test script for the prompt enhancer
echo "🧪 Testing Prompt Enhancer..."

cd /Users/shaansisodia/Desktop/Cursor/SISO_ECOSYSTEM

# Test various prompt types
test_prompts=(
    "fix the bug in login"
    "add a new user dashboard"
    "create a React component for notifications"
    "help me with the database"
    "optimize the performance"
    "can you help me"
)

for prompt in "${test_prompts[@]}"; do
    echo ""
    echo "========================================="
    echo "Original: $prompt"
    echo "========================================="
    
    enhanced=$(~/.claude/scripts/prompt-enhancer.sh "$prompt")
    echo "$enhanced"
    echo ""
    echo "----------------------------------------"
done

echo ""
echo "✅ Test complete! Check ~/.claude/logs/prompt-enhancer.log for details"