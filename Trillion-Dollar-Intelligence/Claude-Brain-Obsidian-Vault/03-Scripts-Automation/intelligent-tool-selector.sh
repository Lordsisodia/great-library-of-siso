#!/bin/bash
# Intelligent Tool Selection System - Removes Claude Code bottlenecks

TASK="$1"
CONTEXT="$2"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 Intelligent Tool Selection Active${NC}"
echo -e "${YELLOW}Analyzing task for optimal tool usage...${NC}"
echo ""

# Function to suggest tools
suggest_tools() {
    local category="$1"
    local tools="$2"
    local reason="$3"
    
    echo -e "${GREEN}📦 $category${NC}"
    echo -e "   ${YELLOW}Tools:${NC} $tools"
    echo -e "   ${YELLOW}Why:${NC} $reason"
    echo ""
}

# Analyze task for patterns
TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')

# Database patterns
if echo "$TASK_LOWER" | grep -qE "(database|sql|query|migration|supabase|postgres)"; then
    suggest_tools "Database Operations" \
        "mcp__supabase__* tools + @agent-laravel-eloquent-expert" \
        "Direct database access with Supabase MCP instead of just writing SQL files"
fi

# Research patterns
if echo "$TASK_LOWER" | grep -qE "(research|investigate|explore|find out|learn about|understand)"; then
    suggest_tools "Research & Discovery" \
        "mcp__exa__deep_researcher_start + mcp__context7-mcp__* + @agent-code-archaeologist" \
        "AI-powered deep research instead of basic web search"
fi

# Documentation patterns
if echo "$TASK_LOWER" | grep -qE "(document|readme|guide|tutorial|explain|notion)"; then
    suggest_tools "Documentation" \
        "mcp__notion__* tools + @agent-documentation-specialist" \
        "Create rich documentation in Notion instead of just markdown files"
fi

# File analysis patterns
if echo "$TASK_LOWER" | grep -qE "(analyze|csv|json|data|pandas|statistics|excel)"; then
    suggest_tools "Data Analysis" \
        "mcp__desktop-commander__start_process (Python REPL) + interact_with_process" \
        "Use interactive Python for real data analysis instead of just reading files"
fi

# Complex project patterns
if echo "$TASK_LOWER" | grep -qE "(build|create|implement|develop|feature|system|application)"; then
    suggest_tools "Complex Development" \
        "@agent-tech-lead-orchestrator + specialist agents + AUTONOMOUS-CLAUDE-SYSTEM" \
        "Leverage full team instead of coding alone"
fi

# Performance patterns
if echo "$TASK_LOWER" | grep -qE "(slow|optimize|performance|speed|bottleneck|profile)"; then
    suggest_tools "Performance Optimization" \
        "@agent-performance-optimizer + mcp__desktop-commander__* profiling" \
        "Professional optimization instead of guesswork"
fi

# Security patterns
if echo "$TASK_LOWER" | grep -qE "(security|vulnerability|auth|password|encrypt|safe)"; then
    suggest_tools "Security Analysis" \
        "@agent-code-reviewer + mcp__supabase__get_advisors" \
        "Professional security audit instead of basic checks"
fi

# Testing patterns
if echo "$TASK_LOWER" | grep -qE "(test|spec|tdd|coverage|unit|integration)"; then
    suggest_tools "Testing Strategy" \
        "@agent-tester + @agent-tdd-london-swarm + mcp__desktop-commander__*" \
        "Comprehensive test generation instead of basic tests"
fi

# Show available MCPs
echo -e "${BLUE}📋 Available MCP Tools Summary:${NC}"
echo "• Supabase: Database operations, migrations, logs"
echo "• Notion: Documentation, knowledge management"
echo "• Exa: Deep research, company research, web scraping"
echo "• Desktop Commander: File ops, process control, code search"
echo "• Puppeteer: Browser automation, screenshots"
echo "• AppleScript: System control, app automation"
echo "• Clear Thought: Chain of thought reasoning"
echo ""

# Show available agents
echo -e "${BLUE}👥 Available Specialist Agents:${NC}"
echo "• Orchestrators: tech-lead, team-configurator, project-analyst"
echo "• Backend: laravel, django, rails experts + universal backend"
echo "• Frontend: react, vue specialists + component architects"
echo "• Core: code-reviewer, performance-optimizer, archaeologist"
echo ""

# Show autonomous systems
echo -e "${BLUE}🤖 Autonomous Systems:${NC}"
echo "• TMUX Teams: Mining, Analysis, Integration, Client teams"
echo "• Telegram Fleet: Real-time notifications and control"
echo "• Memory Systems: Cross-session persistence"
echo ""

# Performance tip
echo -e "${RED}⚡ Performance Tip:${NC}"
echo "Combine multiple tools for maximum effectiveness!"
echo "Example: Research with Exa → Document in Notion → Implement with agents"

# Save suggestions to memory
echo "{
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"task\": \"$TASK\",
  \"suggested_tools\": \"See above\",
  \"context\": \"$CONTEXT\"
}" >> ~/.claude/analytics/tool-suggestions.jsonl