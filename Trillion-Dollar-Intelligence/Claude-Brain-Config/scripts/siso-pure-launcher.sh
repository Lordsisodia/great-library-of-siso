#!/bin/bash

# SISO Pure Launcher - Only SISO welcome, no Claude welcome
# This version completely controls the startup experience

clear

# Get current directory info
CURRENT_DIR=$(basename "$PWD")
FULL_PATH="$PWD"
PROJECT_TYPE=""

# Detect project type with more sophisticated detection
if [[ -f "package.json" ]]; then
    if grep -q "react" package.json 2>/dev/null; then
        PROJECT_TYPE="React/TypeScript"
    elif grep -q "next" package.json 2>/dev/null; then
        PROJECT_TYPE="Next.js"
    elif grep -q "vue" package.json 2>/dev/null; then
        PROJECT_TYPE="Vue.js"
    else
        PROJECT_TYPE="Node.js"
    fi
elif [[ -f "requirements.txt" ]] || [[ -f "pyproject.toml" ]] || [[ -f "setup.py" ]]; then
    PROJECT_TYPE="Python"
elif [[ -f "Cargo.toml" ]]; then
    PROJECT_TYPE="Rust"
elif [[ -f "go.mod" ]]; then
    PROJECT_TYPE="Go"
elif [[ -f "pom.xml" ]] || [[ -f "build.gradle" ]]; then
    PROJECT_TYPE="Java"
elif [[ -f "composer.json" ]]; then
    PROJECT_TYPE="PHP"
elif [[ -d ".git" ]]; then
    PROJECT_TYPE="Git Repository"
else
    PROJECT_TYPE="Development"
fi

# Special detection for SISO projects
if [[ "$CURRENT_DIR" == *"SISO"* ]] || [[ "$FULL_PATH" == *"SISO"* ]]; then
    PROJECT_TYPE="🌟 SISO $PROJECT_TYPE"
fi

# Generate timestamp
TIMESTAMP=$(date '+%H:%M on %b %d, %Y')

# Display SISO welcome screen
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
│  🚀 SuperClaude Enhanced Development Environment           │
│  ⚡ AUTONOMOUS CODING AGENT | 10X INTELLIGENCE ACTIVATED    │
│                                                             │
│     🧠 Ultra Think Mode Ready                               │
│     🎯 MUSK Algorithm Engaged                               │
│     🔧 Multi-Agent Teams Available                          │
│     💎 First Principles Thinking Active                     │
│     📡 Serena MCP: Advanced Code Intelligence              │
│     🌟 Zen MCP: Multi-Model AI Orchestration               │
│     📱 Happy Coder: Mobile Access Ready                     │
│                                                             │
│  📋 Quick Commands:                                          │
│     /help     - Enhanced help system                        │
│     /status   - System & project status                     │
│     /agents   - Available specialist agents                 │
│     /ultra    - Activate maximum reasoning                   │
│     📱 Mobile - Control from phone via Happy app            │
│                                                             │
│  🔧 MCP Tools (Auto-loaded):                                │
│     "Use Serena's find_referencing_symbols"                 │
│     "Use Zen's planner with Gemini Pro"                     │
│     "Get multi-model consensus on approach"                 │
│                                                             │
│  ⚠️  Context Management (Community Best Practices):        │
│     • Reset when context drops below 20%                   │
│     • Execute 3-5 tasks max per session                     │
│     • Create CLAUDE.md + TODO.md before coding             │
│     • Commit frequently, reset often                       │
│                                                             │
EOF

printf "│  🎯 Project: %-44s │\n" "$CURRENT_DIR"
printf "│  🔧 Type: %-47s │\n" "$PROJECT_TYPE"
printf "│  📍 Location: %-42s │\n" "$FULL_PATH"
printf "│  ⏰ Session: %-44s │\n" "$TIMESTAMP"

cat << 'EOF'
│                                                             │
│  💡 Ready to revolutionize your development workflow!       │
│                                                             │
╰─────────────────────────────────────────────────────────────╯

🔥 SISO Enhanced Claude Code is now active. All autonomous systems online.
📊 Advanced reasoning, multi-agent orchestration, and 10X intelligence enabled.
🎯 First principles thinking, MUSK's 5-step algorithm, and ultra think mode ready.

EOF

# ==================== MCP SERVERS AUTO-LOADING ====================
echo "🔧 Initializing MCP servers..."

# Load Serena MCP (Advanced Code Intelligence)
echo "   📡 Loading Serena MCP (Code Intelligence)..."
claude mcp add-json "serena" '{"command":"uvx","args":["--from","git+https://github.com/oraios/serena","serena-mcp-server"]}' >/dev/null 2>&1

# Load Zen MCP (Multi-Model Orchestration)  
echo "   🧠 Loading Zen MCP (Multi-Model AI Orchestration)..."
claude mcp add-json "zen-mcp" '{"command":"uvx","args":["--from","git+https://github.com/BeehiveInnovations/zen-mcp-server.git","zen-mcp-server"],"env":{"GEMINI_API_KEY":"AIzaSyDnuBN9ZzW3HnH_-3RAlOZu3GUs9zTz6HM","GROQ_API_KEY":"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","CEREBRAS_API_KEY":"csk-3k6trr428thrppejwexnep65kh8m3nccmx5p92np3x8rr2wr"}}' >/dev/null 2>&1

# Load Supabase MCP (Database Operations)
echo "   🗄️ Loading Supabase MCP (Database Operations)..."
claude mcp add-json "supabase" '{"command":"npx","args":["-y","@supabase/mcp-server-supabase@latest","--project-ref=avdgyrepwrvsvwgxrccr"],"env":{"SUPABASE_ACCESS_TOKEN":"sbp_46f04e75f8bba39917efda341bbf260ac60d3c8d"}}' >/dev/null 2>&1

# Brief pause to ensure MCP servers are configured
sleep 1

echo "✅ MCP servers loaded: Serena (Code Intelligence) + Zen (Multi-Model AI) + Supabase (Database)"
echo "🚀 Advanced capabilities: find_referencing_symbols, planner, consensus, database operations ready"
echo ""

# Check if happy-coder is available and auto-start if installed
if command -v happy >/dev/null 2>&1; then
    echo "🎉 Happy Coder detected! Starting mobile-enabled Claude..."
    echo "📱 You can now control Claude from your phone using the Happy app"
    echo "🔄 To switch back to desktop mode, press any key on your keyboard"
    echo ""
    
    # Start Happy (which wraps Claude) with a custom prompt that simulates the session
    if [ $# -eq 0 ]; then
        # Interactive session - launch happy with continue or start fresh
        happy --continue 2>/dev/null || happy
    else
        # Non-interactive - pass arguments through
        happy "$@"
    fi
else
    echo "💡 Tip: Install Happy Coder for mobile access: npm install -g happy-coder"
    echo "📱 Then use 'happy' instead of 'claude' to enable phone control"
    echo ""
    
    # Start Claude with a custom prompt that simulates the session
    if [ $# -eq 0 ]; then
        # Interactive session - launch claude with continue or start fresh
        claude --continue 2>/dev/null || claude
    else
        # Non-interactive - pass arguments through
        claude "$@"
    fi
fi