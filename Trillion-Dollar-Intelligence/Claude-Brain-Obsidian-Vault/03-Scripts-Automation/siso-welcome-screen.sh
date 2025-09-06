#!/bin/bash

# SISO-branded welcome screen for Claude Code
# Creates a custom SISO welcome message with dynamic project detection

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

# Create SISO welcome message with dynamic content and ASCII art
cat << EOF
{
  "context": "
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
│                                                             │
│  📋 Quick Commands:                                          │
│     /help     - Enhanced help system                        │
│     /status   - System & project status                     │
│     /agents   - Available specialist agents                 │
│     /ultra    - Activate maximum reasoning                   │
│                                                             │
│  🎯 Project: $CURRENT_DIR                        │
│  🔧 Type: $PROJECT_TYPE                          │
│  📍 Location: $FULL_PATH                               │
│  ⏰ Session: $TIMESTAMP                                    │
│                                                             │
│  💡 Ready to revolutionize your development workflow!       │
│                                                             │
╰─────────────────────────────────────────────────────────────╯

🔥 SISO Enhanced Claude Code is now active. All autonomous systems online.
📊 Advanced reasoning, multi-agent orchestration, and 10X intelligence enabled.
🎯 First principles thinking, MUSK's 5-step algorithm, and ultra think mode ready.

"
}
EOF