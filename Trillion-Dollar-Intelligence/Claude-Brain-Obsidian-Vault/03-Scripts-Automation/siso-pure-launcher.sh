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

# Display clean SISO welcome
echo "   ███████╗██╗███████╗ ██████╗ "
echo "   ██╔════╝██║██╔════╝██╔═══██╗"
echo "   ███████╗██║███████╗██║   ██║"
echo "   ╚════██║██║╚════██║██║   ██║"
echo "   ███████║██║███████║╚██████╔╝"
echo "   ╚══════╝╚═╝╚══════╝ ╚═════╝ "
echo ""
echo "   🚀 SuperClaude Enhanced Development Environment"
echo "   📁 $CURRENT_DIR • $PROJECT_TYPE • $TIMESTAMP"
echo ""

# Load MCP servers quietly (skip if already exist)
claude mcp add-json "serena" '{"command":"uvx","args":["--from","git+https://github.com/oraios/serena","serena-mcp-server"]}' >/dev/null 2>&1 || true
claude mcp add-json "zen-mcp" '{"command":"uvx","args":["--from","git+https://github.com/BeehiveInnovations/zen-mcp-server.git","zen-mcp-server"],"env":{"GEMINI_API_KEY":"AIzaSyDnuBN9ZzW3HnH_-3RAlOZu3GUs9zTz6HM","GROQ_API_KEY":"YOUR_GROQ_API_KEY_HERE","CEREBRAS_API_KEY":"csk-3k6trr428thrppejwexnep65kh8m3nccmx5p92np3x8rr2wr"}}' >/dev/null 2>&1 || true
claude mcp add-json "supabase" '{"command":"npx","args":["-y","@supabase/mcp-server-supabase@latest","--project-ref=avdgyrepwrvsvwgxrccr"],"env":{"SUPABASE_ACCESS_TOKEN":"sbp_46f04e75f8bba39917efda341bbf260ac60d3c8d"}}' >/dev/null 2>&1 || true
echo "✅ MCP servers initialized"

# Enable MCP metadata display for token/model visibility
export MCP_CLAUDE_DEBUG=true
echo "🔍 MCP Debug Mode: ENABLED - Token counts and model routing now visible!"
echo "💡 Look for 'metadata' sections in responses to see model/provider info"
echo ""

# Launch Claude or Happy if available with debug flags
if command -v happy >/dev/null 2>&1; then
    if [ $# -eq 0 ]; then
        happy --mcp-debug --continue 2>/dev/null || happy --mcp-debug
    else
        happy --mcp-debug "$@"
    fi
else
    if [ $# -eq 0 ]; then
        claude --mcp-debug --continue 2>/dev/null || claude --mcp-debug
    else
        claude --mcp-debug "$@"
    fi
fi