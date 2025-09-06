#!/bin/bash

# 🔍 Claude Brain Smart Search System
# Quick search shortcuts for 3,083+ files

BRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAGS_FILE="$BRAIN_DIR/FILE-CATEGORIES.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🔍 Claude Brain Smart Search${NC}"

# Smart search functions
search_intelligence() {
    local query="${1:-}"
    echo -e "${CYAN}🧠 Intelligence Systems Search${NC}"
    if [ -n "$query" ]; then
        echo "Searching for: '$query'"
        find "$BRAIN_DIR/shared" -name "*.yml" -exec grep -l "$query" {} \; | sed "s|$BRAIN_DIR/||"
    else
        echo "Available intelligence modules:"
        find "$BRAIN_DIR/shared" -name "*intelligence*.yml" | sed "s|$BRAIN_DIR/||" | sort
    fi
}

search_agents() {
    local query="${1:-}"
    echo -e "${PURPLE}🤖 Agent Systems Search${NC}"
    if [ -n "$query" ]; then
        echo "Searching for: '$query'"
        find "$BRAIN_DIR/agents" -name "*.md" -exec grep -l "$query" {} \; | sed "s|$BRAIN_DIR/||"
    else
        echo "Available agents:"
        find "$BRAIN_DIR/agents" -name "*.md" | sed "s|$BRAIN_DIR/||; s|agents/||; s|\.md$||" | sort
    fi
}

search_scripts() {
    local query="${1:-}"
    echo -e "${GREEN}⚙️ Automation Scripts Search${NC}"
    if [ -n "$query" ]; then
        echo "Searching for: '$query'"
        find "$BRAIN_DIR/scripts" -name "*.sh" -exec grep -l "$query" {} \; | sed "s|$BRAIN_DIR/||"
    else
        echo "Script categories:"
        echo "  auto-*     : Automation systems"
        echo "  *optimizer*: Performance optimization"  
        echo "  intelligent-*: Smart processing"
        echo "  *monitor*  : System monitoring"
        echo
        echo "Recent scripts:"
        find "$BRAIN_DIR/scripts" -name "*.sh" | head -10 | sed "s|$BRAIN_DIR/||"
    fi
}

search_docs() {
    local query="${1:-}"
    echo -e "${YELLOW}📚 Documentation Search${NC}"
    if [ -n "$query" ]; then
        echo "Searching for: '$query'"
        find "$BRAIN_DIR" -name "*.md" -exec grep -l "$query" {} \; | sed "s|$BRAIN_DIR/||" | head -10
    else
        echo "Key documentation:"
        find "$BRAIN_DIR" \( -name "*README*" -o -name "*GUIDE*" -o -name "*INDEX*" \) -name "*.md" | sed "s|$BRAIN_DIR/||"
    fi
}

search_configs() {
    local query="${1:-}"
    echo -e "${CYAN}⚙️ Configuration Search${NC}"
    if [ -n "$query" ]; then
        echo "Searching for: '$query'"
        find "$BRAIN_DIR" -name "*.yml" -o -name "*.json" | xargs grep -l "$query" 2>/dev/null | sed "s|$BRAIN_DIR/||" | head -10
    else
        echo "Configuration files:"
        echo "  CLAUDE.md           : Main configuration"
        echo "  shared/*.yml        : Intelligence modules"
        echo "  settings*.json      : System settings"
        echo "  analytics/*.json    : Performance data"
    fi
}

search_by_keyword() {
    local keyword="$1"
    local limit="${2:-10}"
    
    echo -e "${BLUE}🎯 Keyword Search: '$keyword'${NC}"
    echo
    
    # Search in different categories
    echo -e "${CYAN}Intelligence modules:${NC}"
    grep -r "$keyword" "$BRAIN_DIR/shared/" --include="*.yml" | head -3 | sed "s|$BRAIN_DIR/||"
    echo
    
    echo -e "${PURPLE}Agent configurations:${NC}"
    grep -r "$keyword" "$BRAIN_DIR/agents/" --include="*.md" | head -3 | sed "s|$BRAIN_DIR/||"
    echo
    
    echo -e "${GREEN}Scripts:${NC}"
    grep -r "$keyword" "$BRAIN_DIR/scripts/" --include="*.sh" | head -3 | sed "s|$BRAIN_DIR/||"
    echo
    
    echo -e "${YELLOW}Documentation:${NC}"
    grep -r "$keyword" "$BRAIN_DIR" --include="*.md" | head -3 | sed "s|$BRAIN_DIR/||"
}

# Quick access patterns
quick_core() {
    echo -e "${CYAN}🎯 Core System Files${NC}"
    echo "CLAUDE.md                                  # Main configuration"
    echo "shared/superclaude-core.yml               # Core intelligence (70% token reduction)"
    echo "shared/musk-algorithm-core.yml            # First principles framework"
    echo "shared/task-intelligence-system.yml       # Task management"
    echo "agents/multi-agent-orchestrator.md        # Agent coordination"
    echo "scripts/claude-universal-enhancer.sh      # Universal enhancement"
    echo "scripts/openai-prompt-optimizer.sh        # Prompt optimization"
}

quick_performance() {
    echo -e "${GREEN}⚡ Performance & Optimization${NC}"
    echo "analytics/enhancement-metrics.csv         # Performance metrics"
    echo "shared/token-economy-intelligence.yml     # Resource optimization"
    echo "shared/compute-optimization-intelligence.yml # Efficiency patterns"
    echo "scripts/daily-improvement-report.sh       # Daily metrics"
    echo "scripts/improvement-tracker.sh            # Progress tracking"
}

quick_automation() {
    echo -e "${PURPLE}🤖 Automation Systems${NC}"
    find "$BRAIN_DIR/scripts" -name "auto-*" | sed "s|$BRAIN_DIR/||" | head -5
    echo "scripts/intelligent-session.sh            # Smart session management"
    echo "scripts/system-activator.sh               # System initialization"
}

# Recent activity
show_recent() {
    local days="${1:-7}"
    echo -e "${YELLOW}📅 Recent Activity (last $days days)${NC}"
    
    echo -e "${CYAN}Modified files:${NC}"
    find "$BRAIN_DIR" -type f \( -name "*.md" -o -name "*.yml" -o -name "*.sh" \) -mtime -$days | sed "s|$BRAIN_DIR/||" | head -10
    
    echo
    echo -e "${CYAN}Recent logs:${NC}"
    find "$BRAIN_DIR/logs" -name "*.log" -mtime -$days | sed "s|$BRAIN_DIR/||" | head -5
    
    echo  
    echo -e "${CYAN}New todos:${NC}"
    find "$BRAIN_DIR/todos" -name "*.json" -mtime -$days | wc -l | xargs echo "Active tasks:"
}

# Interactive search menu
interactive_menu() {
    echo -e "${BLUE}🔍 Interactive Search Menu${NC}"
    echo
    echo "1) Search Intelligence Systems"
    echo "2) Search Agents"  
    echo "3) Search Scripts"
    echo "4) Search Documentation"
    echo "5) Search Configurations"
    echo "6) Keyword Search"
    echo "7) Quick Core Files"
    echo "8) Quick Performance"
    echo "9) Quick Automation"
    echo "10) Recent Activity"
    echo "0) Exit"
    echo
    read -p "Select option: " choice
    
    case $choice in
        1) read -p "Search term (optional): " term; search_intelligence "$term" ;;
        2) read -p "Search term (optional): " term; search_agents "$term" ;;
        3) read -p "Search term (optional): " term; search_scripts "$term" ;;
        4) read -p "Search term (optional): " term; search_docs "$term" ;;
        5) read -p "Search term (optional): " term; search_configs "$term" ;;
        6) read -p "Enter keyword: " term; search_by_keyword "$term" ;;
        7) quick_core ;;
        8) quick_performance ;;
        9) quick_automation ;;
        10) read -p "Days to check (default 7): " days; show_recent "${days:-7}" ;;
        0) exit 0 ;;
        *) echo "Invalid option" ;;
    esac
}

# Usage
show_usage() {
    echo "Usage: $0 [OPTION] [QUERY]"
    echo
    echo "Options:"
    echo "  -i, --intelligence [query]    Search intelligence systems"
    echo "  -a, --agents [query]          Search agent configurations"
    echo "  -s, --scripts [query]         Search automation scripts"
    echo "  -d, --docs [query]            Search documentation"  
    echo "  -c, --configs [query]         Search configuration files"
    echo "  -k, --keyword <query>         Search all files for keyword"
    echo "  -r, --recent [days]           Show recent activity"
    echo "  --core                        Show core system files"
    echo "  --performance                 Show performance files"
    echo "  --automation                  Show automation files"
    echo "  --menu                        Interactive menu"
    echo "  -h, --help                    Show this help"
    echo
    echo "Examples:"
    echo "  $0 --keyword 'optimization'"
    echo "  $0 --agents 'enhanced'"
    echo "  $0 --scripts 'auto'"
    echo "  $0 --recent 3"
    echo "  $0 --menu"
}

# Main execution
case "${1:-}" in
    "-i"|"--intelligence")
        search_intelligence "$2"
        ;;
    "-a"|"--agents")
        search_agents "$2"
        ;;
    "-s"|"--scripts")
        search_scripts "$2"
        ;;
    "-d"|"--docs")
        search_docs "$2"
        ;;
    "-c"|"--configs")
        search_configs "$2"
        ;;
    "-k"|"--keyword")
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Please provide a keyword${NC}"
            exit 1
        fi
        search_by_keyword "$2"
        ;;
    "-r"|"--recent")
        show_recent "${2:-7}"
        ;;
    "--core")
        quick_core
        ;;
    "--performance") 
        quick_performance
        ;;
    "--automation")
        quick_automation
        ;;
    "--menu")
        interactive_menu
        ;;
    "-h"|"--help"|"")
        show_usage
        ;;
    *)
        echo -e "${YELLOW}💡 Quick keyword search for: $1${NC}"
        search_by_keyword "$1"
        ;;
esac