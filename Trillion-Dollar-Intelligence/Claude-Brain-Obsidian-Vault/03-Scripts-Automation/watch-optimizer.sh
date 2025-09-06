#!/bin/bash

# Watch the optimizer in action!

echo "👀 WATCHING OPENAI GPT-5 OPTIMIZER"
echo "=================================="
echo ""
echo "This window will show you:"
echo "1. When prompts are optimized"
echo "2. The transformation details"
echo "3. Visual feedback logs"
echo ""
echo "Keep this window open while using Claude Code!"
echo ""
echo "Press Ctrl+C to stop watching"
echo ""
echo "=================================="
echo ""

# Watch both logs
tail -f ~/.claude/logs/openai-gpt5-optimizer.log ~/.claude/visual-optimizer-output.log | while read line; do
    if [[ "$line" =~ "🚀🚀🚀" ]]; then
        # Clear screen for new optimization
        clear
        echo "👀 OPTIMIZER ACTIVITY DETECTED!"
        echo "=============================="
    fi
    echo "$line"
done