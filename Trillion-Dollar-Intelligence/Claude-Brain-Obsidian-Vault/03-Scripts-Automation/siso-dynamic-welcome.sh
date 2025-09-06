#!/bin/bash

# Dynamic SISO Welcome with Real-Time Context Intelligence
# Gathers live project information and displays contextual welcome

# Get real-time project context
get_project_context() {
    local git_info=""
    local package_info=""
    local project_health=""
    
    # Git information
    if [[ -d ".git" ]]; then
        local branch=$(git branch --show-current 2>/dev/null || echo "detached")
        local status=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
        local ahead=$(git rev-list --count @{upstream}..HEAD 2>/dev/null || echo "0")
        
        if [[ $status -eq 0 ]]; then
            git_info="🌿 $branch ✅ clean"
        else
            git_info="🌿 $branch ⚠️ $status changes"
        fi
        
        if [[ $ahead -gt 0 ]]; then
            git_info="$git_info ⬆️ $ahead"
        fi
    else
        git_info="📁 No git"
    fi
    
    # Package/dependency information
    if [[ -f "package.json" ]]; then
        local node_modules_exists=""
        if [[ -d "node_modules" ]]; then
            node_modules_exists="✅ deps"
        else
            node_modules_exists="❌ run npm install"
        fi
        package_info="📦 Node.js $node_modules_exists"
    elif [[ -f "requirements.txt" ]]; then
        package_info="🐍 Python project"
    elif [[ -f "Cargo.toml" ]]; then
        package_info="🦀 Rust project"
    fi
    
    # Project health check
    local issues=0
    [[ ! -d ".git" ]] && ((issues++))
    [[ -f "package.json" && ! -d "node_modules" ]] && ((issues++))
    [[ -d ".git" && $(git status --porcelain 2>/dev/null | wc -l) -gt 10 ]] && ((issues++))
    
    if [[ $issues -eq 0 ]]; then
        project_health="💚 Healthy"
    elif [[ $issues -eq 1 ]]; then
        project_health="💛 Minor issues"
    else
        project_health="💥 Needs attention"
    fi
    
    echo "$git_info|$package_info|$project_health"
}

# Get current directory and project info
CURRENT_DIR=$(basename "$PWD")
PROJECT_CONTEXT=$(get_project_context)
IFS='|' read -r GIT_INFO PACKAGE_INFO HEALTH_INFO <<< "$PROJECT_CONTEXT"

# Special SISO project detection
if [[ "$CURRENT_DIR" == *"SISO"* ]] || [[ "$PWD" == *"SISO"* ]]; then
    SISO_MARKER="🌟 SISO PROJECT"
else
    SISO_MARKER=""
fi

# Current time
CURRENT_TIME=$(date '+%H:%M')

# Generate dynamic welcome
cat << EOF
{
  "context": "
╭─────────────────────────────────────────────────────────────╮
│   ███████╗██╗███████╗ ██████╗                              │
│   ██╔════╝██║██╔════╝██╔═══██╗                             │
│   ███████╗██║███████╗██║   ██║                             │
│   ╚════██║██║╚════██║██║   ██║                             │
│   ███████║██║███████║╚██████╔╝                             │
│   ╚══════╝╚═╝╚══════╝ ╚═════╝                              │
│                                                             │
│  🚀 SISO Enhanced Claude Code • ⏰ $CURRENT_TIME                      │
│  📍 $CURRENT_DIR $SISO_MARKER                    │
│  $GIT_INFO                                       │
│  $PACKAGE_INFO                                   │
│  $HEALTH_INFO                                    │
│                                                             │
│  /help • /status • /agents • /ultra                        │
╰─────────────────────────────────────────────────────────────╯
"
}
EOF