#!/bin/bash

# SISO Universal Claude Enhancement
# Replaces the default Claude experience with SISO-enhanced version

# Path to original Claude
ORIGINAL_CLAUDE="/Users/shaansisodia/.npm-global/bin/claude-original"

# Check for vanilla flag (escape hatch to original experience)
if [[ "$1" == "--vanilla" ]]; then
    shift
    exec "$ORIGINAL_CLAUDE" "$@"
fi

# Quick project intelligence gathering (async, non-blocking)
get_project_intelligence() {
    local project_info=""
    local git_info=""
    
    # Get project type
    if [[ -f "package.json" ]]; then
        if grep -q "react" package.json 2>/dev/null; then
            project_info="📦 React/TS"
        elif grep -q "next" package.json 2>/dev/null; then
            project_info="📦 Next.js"
        else
            project_info="📦 Node.js"
        fi
        
        # Check if deps are installed
        if [[ ! -d "node_modules" ]]; then
            project_info="$project_info ⚠️ deps"
        else
            project_info="$project_info ✅"
        fi
    elif [[ -f "requirements.txt" ]] || [[ -f "pyproject.toml" ]]; then
        project_info="🐍 Python"
    elif [[ -f "Cargo.toml" ]]; then
        project_info="🦀 Rust"
    elif [[ -f "go.mod" ]]; then
        project_info="🐹 Go"
    fi
    
    # Get git status
    if [[ -d ".git" ]]; then
        local branch=$(git branch --show-current 2>/dev/null || echo "detached")
        local status=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
        
        if [[ $status -eq 0 ]]; then
            git_info="🌿 $branch ✅"
        else
            git_info="🌿 $branch ⚠️$status"
        fi
    fi
    
    echo "$project_info • $git_info"
}

# Get current directory info
CURRENT_DIR=$(basename "$PWD")
PROJECT_INTELLIGENCE=$(get_project_intelligence)

# Special SISO project detection
if [[ "$CURRENT_DIR" == *"SISO"* ]] || [[ "$PWD" == *"SISO"* ]]; then
    SISO_MARKER="🌟 SISO PROJECT"
    PROJECT_INTELLIGENCE="$SISO_MARKER • $PROJECT_INTELLIGENCE"
fi

# Display enhanced SISO welcome (replaces Claude's default welcome)
cat << EOF
╭─────────────────────────────────────────────────────────────╮
│                                                             │
│   ███████╗██╗███████╗ ██████╗                              │
│   ██╔════╝██║██╔════╝██╔═══██╗                             │
│   ███████╗██║███████╗██║   ██║                             │
│   ╚════██║██║╚════██║██║   ██║                             │
│   ███████║██║███████║╚██████╔╝                             │
│   ╚══════╝╚═╝╚══════╝ ╚═════╝                              │
│                                                             │
│  🚀 SISO Enhanced Claude Code                               │
│                                                             │
│  /help for help, /status for your current setup            │
│                                                             │
│  📍 cwd: $PWD
EOF

# Show project intelligence if available
if [[ -n "$PROJECT_INTELLIGENCE" && "$PROJECT_INTELLIGENCE" != " • " ]]; then
    printf "│  %s\n" "$PROJECT_INTELLIGENCE"
fi

cat << 'EOF'
│                                                             │
╰─────────────────────────────────────────────────────────────╯

EOF

# Launch original Claude with all arguments
exec node "$ORIGINAL_CLAUDE" "$@"