#!/bin/bash

# SISO Status Line - Enhanced status information for Claude Code
# Provides real-time project context and SISO branding

# Color codes
SISO_BLUE='\033[36m'
SISO_GREEN='\033[32m'
SISO_YELLOW='\033[33m'
SISO_RED='\033[31m'
SISO_PURPLE='\033[35m'
SISO_BOLD='\033[1m'
RESET='\033[0m'

# Get current directory and project info
CURRENT_DIR=$(basename "$PWD")
PROJECT_STATUS=""
GIT_STATUS=""
HEALTH_INDICATOR=""

# SISO project detection
if [[ "$CURRENT_DIR" == *"SISO"* ]] || [[ "$PWD" == *"SISO"* ]]; then
    SISO_INDICATOR="${SISO_PURPLE}🌟 SISO${RESET}"
else
    SISO_INDICATOR=""
fi

# Project type detection
if [[ -f "package.json" ]]; then
    if grep -q "react" package.json 2>/dev/null; then
        PROJECT_STATUS="${SISO_BLUE}⚛️ React${RESET}"
    elif grep -q "next" package.json 2>/dev/null; then
        PROJECT_STATUS="${SISO_BLUE}🔺 Next.js${RESET}"
    elif grep -q "vue" package.json 2>/dev/null; then
        PROJECT_STATUS="${SISO_GREEN}💚 Vue${RESET}"
    else
        PROJECT_STATUS="${SISO_YELLOW}📦 Node${RESET}"
    fi
    
    # Check dependencies
    if [[ ! -d "node_modules" ]]; then
        PROJECT_STATUS="$PROJECT_STATUS ${SISO_RED}⚠️${RESET}"
    fi
elif [[ -f "requirements.txt" ]] || [[ -f "pyproject.toml" ]]; then
    PROJECT_STATUS="${SISO_GREEN}🐍 Python${RESET}"
elif [[ -f "Cargo.toml" ]]; then
    PROJECT_STATUS="${SISO_RED}🦀 Rust${RESET}"
elif [[ -f "go.mod" ]]; then
    PROJECT_STATUS="${SISO_BLUE}🐹 Go${RESET}"
fi

# Git status
if [[ -d ".git" ]]; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "detached")
    CHANGES=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    
    if [[ $CHANGES -eq 0 ]]; then
        GIT_STATUS="${SISO_GREEN}🌿 $BRANCH ✅${RESET}"
    else
        GIT_STATUS="${SISO_YELLOW}🌿 $BRANCH ⚠️$CHANGES${RESET}"
    fi
    
    # Check if ahead of remote
    AHEAD=$(git rev-list --count @{upstream}..HEAD 2>/dev/null || echo "0")
    if [[ $AHEAD -gt 0 ]]; then
        GIT_STATUS="$GIT_STATUS ${SISO_PURPLE}⬆️$AHEAD${RESET}"
    fi
fi

# Overall health indicator
ISSUES=0
[[ ! -d ".git" ]] && ((ISSUES++))
[[ -f "package.json" && ! -d "node_modules" ]] && ((ISSUES++))
[[ -d ".git" && $(git status --porcelain 2>/dev/null | wc -l) -gt 10 ]] && ((ISSUES++))

if [[ $ISSUES -eq 0 ]]; then
    HEALTH_INDICATOR="${SISO_GREEN}💚${RESET}"
elif [[ $ISSUES -eq 1 ]]; then
    HEALTH_INDICATOR="${SISO_YELLOW}💛${RESET}"
else
    HEALTH_INDICATOR="${SISO_RED}💥${RESET}"
fi

# Current time
TIME=$(date '+%H:%M')

# Build status line
STATUS_LINE="${SISO_BOLD}${SISO_BLUE}SISO${RESET}"

if [[ -n "$SISO_INDICATOR" ]]; then
    STATUS_LINE="$STATUS_LINE $SISO_INDICATOR"
fi

if [[ -n "$PROJECT_STATUS" ]]; then
    STATUS_LINE="$STATUS_LINE $PROJECT_STATUS"
fi

if [[ -n "$GIT_STATUS" ]]; then
    STATUS_LINE="$STATUS_LINE $GIT_STATUS"
fi

STATUS_LINE="$STATUS_LINE $HEALTH_INDICATOR ${SISO_PURPLE}⏰ $TIME${RESET}"

echo -e "$STATUS_LINE"