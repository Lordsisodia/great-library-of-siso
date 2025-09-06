#!/bin/bash
# Auto Team Spawner - Automatically determines when to spawn agent teams

PROMPT="$1"
PROJECT_DIR="$2"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🤖 Auto Team Spawner Analysis${NC}"

# Analyze task complexity
WORD_COUNT=$(echo "$PROMPT" | wc -w)
PROMPT_LOWER=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')

# Count complexity indicators
COMPLEXITY_SCORE=0

# Check for multiple domains/technologies
if echo "$PROMPT_LOWER" | grep -qE "(frontend.*backend|api.*database|auth.*ui)"; then
    COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 3))
    echo "• Multi-domain project detected (+3)"
fi

# Check for system words
SYSTEM_KEYWORDS="system application platform service architecture framework"
for keyword in $SYSTEM_KEYWORDS; do
    if echo "$PROMPT_LOWER" | grep -q "$keyword"; then
        COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 2))
        echo "• System-level keyword '$keyword' (+2)"
        break
    fi
done

# Check for implementation scope
SCOPE_KEYWORDS="complete full entire comprehensive end-to-end production"
for keyword in $SCOPE_KEYWORDS; do
    if echo "$PROMPT_LOWER" | grep -q "$keyword"; then
        COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 2))
        echo "• Large scope keyword '$keyword' (+2)"
        break
    fi
done

# Check word count (longer = more complex)
if [ "$WORD_COUNT" -gt 20 ]; then
    COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 3))
    echo "• Long description ($WORD_COUNT words) (+3)"
elif [ "$WORD_COUNT" -gt 10 ]; then
    COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 1))
    echo "• Medium description ($WORD_COUNT words) (+1)"
fi

# Check for multiple requirements
REQUIREMENT_COUNT=$(echo "$PROMPT_LOWER" | grep -o -E "(and|with|plus|also|including)" | wc -l)
if [ "$REQUIREMENT_COUNT" -gt 2 ]; then
    COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + 2))
    echo "• Multiple requirements ($REQUIREMENT_COUNT connectors) (+2)"
fi

# Detect technology stack
TECH_COUNT=0
TECH_STACK=""

if echo "$PROMPT_LOWER" | grep -qE "(react|vue|angular|frontend)"; then
    TECH_COUNT=$((TECH_COUNT + 1))
    TECH_STACK="$TECH_STACK frontend"
fi

if echo "$PROMPT_LOWER" | grep -qE "(api|backend|server|node|express|django|laravel)"; then
    TECH_COUNT=$((TECH_COUNT + 1))
    TECH_STACK="$TECH_STACK backend"
fi

if echo "$PROMPT_LOWER" | grep -qE "(database|sql|postgres|mysql|mongo)"; then
    TECH_COUNT=$((TECH_COUNT + 1))
    TECH_STACK="$TECH_STACK database"
fi

if echo "$PROMPT_LOWER" | grep -qE "(auth|authentication|login|security)"; then
    TECH_COUNT=$((TECH_COUNT + 1))
    TECH_STACK="$TECH_STACK auth"
fi

if echo "$PROMPT_LOWER" | grep -qE "(test|testing|tdd|spec)"; then
    TECH_COUNT=$((TECH_COUNT + 1))
    TECH_STACK="$TECH_STACK testing"
fi

COMPLEXITY_SCORE=$((COMPLEXITY_SCORE + TECH_COUNT))
echo "• Technology domains: $TECH_COUNT ($TECH_STACK) (+$TECH_COUNT)"

echo ""
echo -e "${YELLOW}📊 Complexity Score: $COMPLEXITY_SCORE${NC}"

# Determine team recommendation
if [ "$COMPLEXITY_SCORE" -ge 8 ]; then
    echo -e "${RED}🚨 HIGH COMPLEXITY DETECTED${NC}"
    echo -e "${GREEN}🤖 RECOMMENDATION: Spawn LARGE agent team (6-8 agents)${NC}"
    echo ""
    echo "Suggested team composition:"
    echo "├── @agent-tech-lead-orchestrator (coordination)"
    echo "├── @agent-system-architect (design)"
    echo "├── @agent-backend-developer-mcp-enhanced (backend)"
    echo "├── @agent-react-component-architect (frontend)" 
    echo "├── @agent-code-reviewer (quality)"
    echo "├── @agent-performance-optimizer (optimization)"
    echo "├── @agent-tester (testing)"
    echo "└── @agent-documentation-specialist (docs)"
    
elif [ "$COMPLEXITY_SCORE" -ge 5 ]; then
    echo -e "${YELLOW}⚠️ MEDIUM COMPLEXITY DETECTED${NC}"
    echo -e "${GREEN}🤖 RECOMMENDATION: Spawn MEDIUM agent team (4-5 agents)${NC}"
    echo ""
    echo "Suggested team composition:"
    echo "├── @agent-tech-lead-orchestrator (coordination)"
    echo "├── Domain specialist based on: $TECH_STACK"
    echo "├── @agent-code-reviewer (quality)"
    echo "└── @agent-tester (testing)"
    
elif [ "$COMPLEXITY_SCORE" -ge 3 ]; then
    echo -e "${BLUE}ℹ️ MODERATE COMPLEXITY DETECTED${NC}"
    echo -e "${GREEN}🤖 RECOMMENDATION: Spawn SMALL agent team (2-3 agents)${NC}"
    echo ""
    echo "Suggested team composition:"
    echo "├── Domain specialist for: $TECH_STACK"
    echo "└── @agent-code-reviewer (quality)"
    
else
    echo -e "${GREEN}✅ LOW COMPLEXITY DETECTED${NC}"
    echo -e "${BLUE}🤖 RECOMMENDATION: Single specialist agent or solo work${NC}"
    echo ""
    echo "Consider using specialist agent for: $TECH_STACK"
fi

echo ""
echo -e "${BLUE}💡 AUTO-SPAWN TRIGGER:${NC}"
if [ "$COMPLEXITY_SCORE" -ge 5 ]; then
    echo "✅ ACTIVATED - Complexity score ≥ 5 triggers automatic team spawning"
    echo "🚀 Preparing to spawn agents in parallel..."
else
    echo "⏸️ NOT ACTIVATED - Complexity score < 5, team spawning optional"
fi

# Log analysis
echo "{
  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"prompt\": \"$PROMPT\",
  \"complexity_score\": $COMPLEXITY_SCORE,
  \"word_count\": $WORD_COUNT,
  \"tech_stack\": \"$TECH_STACK\",
  \"team_recommendation\": \"$([ $COMPLEXITY_SCORE -ge 8 ] && echo 'large' || [ $COMPLEXITY_SCORE -ge 5 ] && echo 'medium' || [ $COMPLEXITY_SCORE -ge 3 ] && echo 'small' || echo 'solo')\",
  \"auto_spawn\": $([ $COMPLEXITY_SCORE -ge 5 ] && echo 'true' || echo 'false')
}" >> ~/.claude/analytics/team-spawn-analysis.jsonl