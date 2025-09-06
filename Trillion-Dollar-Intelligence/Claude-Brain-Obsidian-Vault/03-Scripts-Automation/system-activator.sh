#!/bin/bash
# System Activator - Forces Claude to use ALL available sophisticated systems

PROMPT="$1"
PROJECT_DIR="$2"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🚀 SYSTEM ACTIVATOR - FORCING FULL POWER USAGE${NC}"
echo ""

PROMPT_LENGTH=${#PROMPT}
PROMPT_WORDS=$(echo "$PROMPT" | wc -w)

# 1. GPT-5 Prompt Optimizer Check
if [ "$PROMPT_LENGTH" -gt 100 ] || [ "$PROMPT_WORDS" -gt 20 ]; then
    echo -e "${RED}🧠 MANDATORY: GPT-5 Prompt Optimization${NC}"
    echo "   Prompt length: $PROMPT_LENGTH chars, $PROMPT_WORDS words"
    echo "   🔄 Auto-activating: ~/.claude/scripts/openai-prompt-optimizer-gpt5-visual.sh"
    echo ""
fi

# 2. Intelligent Session Mode Detection
echo -e "${BLUE}🎯 MANDATORY: Intelligent Session Mode${NC}"
if echo "$PROMPT" | grep -qiE "(fix|bug|error|debug)"; then
    echo "   📊 Session Type: DEBUG_MODE"
    echo "DEBUG_MODE=true" > ~/.claude/session-context
elif echo "$PROMPT" | grep -qiE "(feature|add|create|build|implement)"; then
    echo "   ✨ Session Type: FEATURE_MODE" 
    echo "FEATURE_MODE=true" > ~/.claude/session-context
elif echo "$PROMPT" | grep -qiE "(refactor|optimize|improve|clean)"; then
    echo "   🔧 Session Type: REFACTOR_MODE"
    echo "REFACTOR_MODE=true" > ~/.claude/session-context
elif echo "$PROMPT" | grep -qiE "(test|testing|spec|tdd)"; then
    echo "   🧪 Session Type: TEST_MODE"
    echo "TEST_MODE=true" > ~/.claude/session-context
else
    echo "   🎲 Session Type: GENERAL_MODE"
    echo "GENERAL_MODE=true" > ~/.claude/session-context
fi
echo "   🔄 Auto-activating: ~/.claude/scripts/intelligent-session.sh"
echo ""

# 3. AUTONOMOUS-CLAUDE-SYSTEM Dispatch Check
COMPLEXITY_SCORE=0

# Calculate complexity for auto-dispatch
if echo "$PROMPT" | grep -qiE "(system|platform|application|complete|full|entire)"; then
    COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 3))
fi

if [ "$PROMPT_WORDS" -gt 25 ]; then
    COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 2))
fi

if echo "$PROMPT" | grep -qE "(and|with|plus|also|including)" | wc -l | awk '{if($1>2) print 1; else print 0}' | grep -q 1; then
    COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 2))
fi

if [ "$COMPLEXITY_SCORE" -ge 5 ]; then
    echo -e "${RED}🤖 MANDATORY: AUTONOMOUS-CLAUDE-SYSTEM Dispatch${NC}"
    echo "   Complexity Score: $COMPLEXITY_SCORE (≥5 triggers auto-dispatch)"
    echo "   🚀 Auto-activating: TMUX Autonomous Teams"
    echo "   📁 Location: ~/Desktop/Cursor/claude-improvement/AUTONOMOUS-CLAUDE-SYSTEM/"
    echo ""
fi

# 4. Advanced Hooks Activation
echo -e "${GREEN}🔗 MANDATORY: Advanced Hooks Activation${NC}"
echo "   📝 Auto-documentation generator"
echo "   📊 Improvement tracker"
echo "   👀 Multi-agent observer"
echo "   🔊 Contextual audio feedback"
echo "   📱 Telegram notifications"
echo ""

# 5. MCP Tools Forced Usage
echo -e "${YELLOW}🛠️ MANDATORY: MCP Tools Usage${NC}"
if echo "$PROMPT" | grep -qiE "(database|sql|query|table|users|data)"; then
    echo "   🗄️ Database detected → Use mcp__supabase__* tools"
fi

if echo "$PROMPT" | grep -qiE "(research|investigate|learn|study|find out)"; then
    echo "   🔍 Research detected → Use mcp__exa__deep_researcher_start"
fi

if echo "$PROMPT" | grep -qiE "(document|readme|guide|tutorial|explain)"; then
    echo "   📚 Documentation detected → Use mcp__notion__* tools"
fi

if echo "$PROMPT" | grep -qiE "(analyze.*file|csv|json|data.*analysis)"; then
    echo "   📈 File analysis detected → Use mcp__desktop-commander__start_process"
fi
echo ""

# 6. TodoWrite & Task Coordination
echo -e "${BLUE}📋 MANDATORY: TodoWrite & Task Coordination${NC}"
echo "   ⚠️ NEVER use single TodoWrite calls - Always batch 5-10+ todos"
echo "   ⚠️ NEVER spawn agents sequentially - Always parallel in ONE message"
echo "   ✅ Use Task coordination with claude-flow hooks"
echo ""

# 7. Performance Systems
echo -e "${PURPLE}⚡ MANDATORY: Performance Systems${NC}"
echo "   🏗️ Build optimizer"
echo "   🎨 Auto-formatter"
echo "   🧪 Auto-test runner"
echo "   📊 Performance monitoring"
echo ""

# 8. Session Memory
echo -e "${GREEN}💾 MANDATORY: Session Memory${NC}"
echo "   📝 Session logging active"
echo "   🧠 Context persistence enabled"
echo "   📊 Performance tracking"
echo ""

# Summary
echo -e "${RED}🎯 ACTIVATION SUMMARY:${NC}"
echo "✅ ALL sophisticated systems are now MANDATORY"
echo "✅ Claude MUST use these instead of basic tools"
echo "✅ Full power utilization ENFORCED"
echo ""
echo -e "${PURPLE}🚀 YOUR CLAUDE IS NOW OPERATING AT MAXIMUM POWER!${NC}"

# Log activation
echo "{
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"prompt_length\": $PROMPT_LENGTH,
  \"prompt_words\": $PROMPT_WORDS,
  \"complexity_score\": $COMPLEXITY_SCORE,
  \"systems_activated\": \"all\",
  \"forced_power_mode\": true
}" >> ~/.claude/analytics/system-activations.jsonl