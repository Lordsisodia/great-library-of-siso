#!/bin/bash
# Smart Command Interceptor - Redirects basic commands to powerful alternatives

LAST_PROMPT="$1"
WORKING_DIR="$2"

# Extract potential commands from prompt
PROMPT_LOWER=$(echo "$LAST_PROMPT" | tr '[:upper:]' '[:lower:]')

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔄 Smart Command Interceptor${NC}"

# Track redirections
REDIRECTED=false

# Database query patterns
if echo "$PROMPT_LOWER" | grep -qE "(show.*users|select.*from|count.*where|find.*in.*table)"; then
    echo -e "${YELLOW}🔄 INTERCEPTED:${NC} Database query detected"
    echo -e "${GREEN}💡 REDIRECT TO:${NC} mcp__supabase__execute_sql instead of writing SQL files"
    echo -e "${RED}⚡ INTELLIGENCE BOOST:${NC} Direct database access > File operations"
    REDIRECTED=true
fi

# File analysis patterns
if echo "$PROMPT_LOWER" | grep -qE "(analyze.*\.csv|what.*in.*\.json|parse.*data|statistics.*file)"; then
    echo -e "${YELLOW}🔄 INTERCEPTED:${NC} File analysis detected"
    echo -e "${GREEN}💡 REDIRECT TO:${NC} mcp__desktop-commander__start_process(python3 -i)"
    echo -e "${RED}⚡ INTELLIGENCE BOOST:${NC} Interactive analysis > Static file reading"
    REDIRECTED=true
fi

# Research patterns
if echo "$PROMPT_LOWER" | grep -qE "(how does.*work|best practices|research.*approach|learn about|investigate)"; then
    echo -e "${YELLOW}🔄 INTERCEPTED:${NC} Research task detected"
    echo -e "${GREEN}💡 REDIRECT TO:${NC} mcp__exa__deep_researcher_start"
    echo -e "${RED}⚡ INTELLIGENCE BOOST:${NC} AI research > Basic web search"
    REDIRECTED=true
fi

# Complex build patterns (word count > 10 + build keywords)
WORD_COUNT=$(echo "$LAST_PROMPT" | wc -w)
if [ "$WORD_COUNT" -gt 10 ] && echo "$PROMPT_LOWER" | grep -qE "(build.*system|create.*api|implement.*feature|develop.*app)"; then
    echo -e "${YELLOW}🔄 INTERCEPTED:${NC} Complex build project detected"
    echo -e "${GREEN}💡 REDIRECT TO:${NC} @agent-tech-lead-orchestrator + specialist team"
    echo -e "${RED}⚡ INTELLIGENCE BOOST:${NC} Agent team > Solo development"
    REDIRECTED=true
fi

# Documentation patterns
if echo "$PROMPT_LOWER" | grep -qE "(write.*docs|create.*readme|document.*api|explain.*system)"; then
    echo -e "${YELLOW}🔄 INTERCEPTED:${NC} Documentation task detected"
    echo -e "${GREEN}💡 REDIRECT TO:${NC} mcp__notion__create-page"
    echo -e "${RED}⚡ INTELLIGENCE BOOST:${NC} Rich Notion docs > Basic markdown"
    REDIRECTED=true
fi

# Performance analysis patterns
if echo "$PROMPT_LOWER" | grep -qE "(slow|optimize|performance|bottleneck|profile|speed up)"; then
    echo -e "${YELLOW}🔄 INTERCEPTED:${NC} Performance task detected"
    echo -e "${GREEN}💡 REDIRECT TO:${NC} @agent-performance-optimizer + profiling tools"
    echo -e "${RED}⚡ INTELLIGENCE BOOST:${NC} Professional optimization > Manual tuning"
    REDIRECTED=true
fi

# Security audit patterns  
if echo "$PROMPT_LOWER" | grep -qE "(security|vulnerability|audit|safe|encrypt|auth)"; then
    echo -e "${YELLOW}🔄 INTERCEPTED:${NC} Security task detected"
    echo -e "${GREEN}💡 REDIRECT TO:${NC} @agent-code-reviewer + mcp__supabase__get_advisors"
    echo -e "${RED}⚡ INTELLIGENCE BOOST:${NC} Security expert > Basic checks"
    REDIRECTED=true
fi

if [ "$REDIRECTED" = true ]; then
    echo ""
    echo -e "${BLUE}📈 INTELLIGENCE MULTIPLIER ACTIVATED${NC}"
    echo "Instead of basic tools, using SUPERPOWERS! 🚀"
    
    # Log the interception
    echo "{
      \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
      \"original_prompt\": \"$LAST_PROMPT\",
      \"redirections_applied\": true,
      \"intelligence_boost\": \"activated\"
    }" >> ~/.claude/analytics/command-interceptions.jsonl
else
    echo -e "${GREEN}✅ Prompt analysis complete - optimal tools already in use${NC}"
fi

echo ""