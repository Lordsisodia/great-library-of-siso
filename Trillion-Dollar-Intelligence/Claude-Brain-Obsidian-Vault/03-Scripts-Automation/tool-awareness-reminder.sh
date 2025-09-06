#!/bin/bash
# Tool Awareness Reminder - Prevents Claude Code from using basic tools when superpowers exist

PROMPT="$1"
PROMPT_LOWER=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')

# Quick reminders based on task type
REMINDER=""

# Database work
if echo "$PROMPT_LOWER" | grep -qE "(database|sql|migration|table|query)"; then
    REMINDER="💡 Remember: You have mcp__supabase__* tools for direct database operations!"
fi

# File analysis
if echo "$PROMPT_LOWER" | grep -qE "(analyze.*file|csv|json|data analysis|pandas)"; then
    REMINDER="💡 Remember: Use mcp__desktop-commander__start_process for Python REPL data analysis!"
fi

# Research tasks
if echo "$PROMPT_LOWER" | grep -qE "(research|find out|investigate|learn about)"; then
    REMINDER="💡 Remember: Use mcp__exa__deep_researcher_start for AI-powered research!"
fi

# Documentation
if echo "$PROMPT_LOWER" | grep -qE "(document|readme|guide|write docs)"; then
    REMINDER="💡 Remember: Use mcp__notion__* tools to create rich documentation!"
fi

# Complex features
if echo "$PROMPT_LOWER" | grep -qE "(build.*system|create.*feature|implement.*service)"; then
    REMINDER="💡 Remember: Use @agent-tech-lead-orchestrator and specialist agents!"
fi

# Output reminder if applicable
if [ -n "$REMINDER" ]; then
    echo ""
    echo "$REMINDER"
    echo "🧠 You have 50+ MCP tools, 24 specialist agents, and autonomous systems!"
    echo "📚 Check CLAUDE-TOOL-INTELLIGENCE.md for your full arsenal"
    echo ""
fi

# Always show quick stats
TOTAL_MCPS=$(echo "Supabase(10) + Notion(10) + Exa(6) + Desktop(15) + Others(10)" | bc 2>/dev/null || echo "50+")
echo "⚡ Your Power Level: $TOTAL_MCPS MCP tools | 24 agents | ∞ possibilities"