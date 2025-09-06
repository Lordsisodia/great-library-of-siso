#!/bin/bash
# Auto-detect suitable agents based on task analysis

PROMPT="$1"
PROJECT_DIR="${2:-$PWD}"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Autonomous Agent Detection Active${NC}"

# Function to suggest agent
suggest_agent() {
    local agent="$1"
    local reason="$2"
    echo -e "${GREEN}💡 Recommended Agent:${NC} @agent-${agent}"
    echo -e "${YELLOW}   Reason:${NC} ${reason}"
    echo ""
}

# Analyze prompt for task patterns
PROMPT_LOWER=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')

# API/Backend patterns
if echo "$PROMPT_LOWER" | grep -qE "(api|endpoint|rest|graphql|route|controller|webhook)"; then
    suggest_agent "api-architect" "API/endpoint development detected"
fi

# Frontend component patterns
if echo "$PROMPT_LOWER" | grep -qE "(component|react|vue|angular|frontend|ui|ux|design|button|form|modal)"; then
    if [ -f "$PROJECT_DIR/package.json" ] && grep -q "react" "$PROJECT_DIR/package.json"; then
        suggest_agent "react-component-architect" "React component development detected"
    elif [ -f "$PROJECT_DIR/package.json" ] && grep -q "vue" "$PROJECT_DIR/package.json"; then
        suggest_agent "vue-component-architect" "Vue component development detected"
    else
        suggest_agent "frontend-developer" "Frontend development detected"
    fi
fi

# Database patterns
if echo "$PROMPT_LOWER" | grep -qE "(database|migration|query|sql|orm|eloquent|activerecord|schema|table)"; then
    if [ -f "$PROJECT_DIR/composer.json" ] && grep -q "laravel" "$PROJECT_DIR/composer.json"; then
        suggest_agent "laravel-eloquent-expert" "Laravel Eloquent ORM detected"
    elif [ -f "$PROJECT_DIR/requirements.txt" ] && grep -q "django" "$PROJECT_DIR/requirements.txt"; then
        suggest_agent "django-orm-expert" "Django ORM detected"
    elif [ -f "$PROJECT_DIR/Gemfile" ] && grep -q "rails" "$PROJECT_DIR/Gemfile"; then
        suggest_agent "rails-activerecord-expert" "Rails ActiveRecord detected"
    else
        suggest_agent "backend-developer-mcp-enhanced" "Database operations with MCP support"
    fi
fi

# Authentication patterns
if echo "$PROMPT_LOWER" | grep -qE "(auth|authentication|login|jwt|oauth|session|password|security)"; then
    suggest_agent "backend-developer-mcp-enhanced" "Authentication system with Supabase MCP capabilities"
fi

# Code review patterns
if echo "$PROMPT_LOWER" | grep -qE "(review|audit|security|vulnerability|code quality|refactor|clean)"; then
    suggest_agent "code-reviewer" "Code review and quality analysis"
fi

# Performance patterns
if echo "$PROMPT_LOWER" | grep -qE "(performance|optimize|speed|slow|bottleneck|memory|cpu|profile)"; then
    suggest_agent "performance-optimizer" "Performance optimization required"
fi

# Documentation patterns
if echo "$PROMPT_LOWER" | grep -qE "(document|readme|docs|api docs|swagger|openapi|guide|tutorial)"; then
    suggest_agent "documentation-specialist" "Documentation task detected"
fi

# Unknown codebase patterns
if echo "$PROMPT_LOWER" | grep -qE "(explore|understand|analyze|legacy|unfamiliar|what does|how does|structure)"; then
    suggest_agent "code-archaeologist" "Codebase exploration and analysis"
fi

# Complex multi-domain tasks
if echo "$PROMPT_LOWER" | grep -qE "(build.*system|create.*application|develop.*feature|implement.*platform)"; then
    suggest_agent "tech-lead-orchestrator" "Complex multi-domain task requiring orchestration"
fi

# Framework detection from filesystem
echo -e "${BLUE}📦 Detected Stack:${NC}"
if [ -f "$PROJECT_DIR/package.json" ]; then
    if grep -q "next" "$PROJECT_DIR/package.json" 2>/dev/null; then
        echo "  - Next.js → Consider @agent-react-nextjs-expert"
    fi
    if grep -q "nuxt" "$PROJECT_DIR/package.json" 2>/dev/null; then
        echo "  - Nuxt.js → Consider @agent-vue-nuxt-expert"
    fi
fi

if [ -f "$PROJECT_DIR/composer.json" ] && grep -q "laravel" "$PROJECT_DIR/composer.json" 2>/dev/null; then
    echo "  - Laravel → Consider @agent-laravel-backend-expert"
fi

if [ -f "$PROJECT_DIR/requirements.txt" ] && grep -q "django" "$PROJECT_DIR/requirements.txt" 2>/dev/null; then
    echo "  - Django → Consider @agent-django-backend-expert"
fi

# Agent creation suggestion
WORDS=$(echo "$PROMPT" | wc -w)
if [ "$WORDS" -gt 20 ]; then
    echo ""
    echo -e "${YELLOW}📝 Complex task detected.${NC} If no perfect agent exists, I can create a specialized one."
fi

# Save detection for analytics
mkdir -p ~/.claude/analytics
echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"prompt_length\":$WORDS,\"detected_patterns\":\"$PROMPT_LOWER\"}" >> ~/.claude/analytics/agent-detection.jsonl