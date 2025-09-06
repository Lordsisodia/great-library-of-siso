#!/bin/bash

# SISO Claude Code Launcher with custom welcome screen
# This script shows the SISO welcome screen and keeps it visible

# Create settings file to suppress Claude's welcome screen
SISO_SETTINGS_FILE="/tmp/siso-claude-settings.json"
cat > "$SISO_SETTINGS_FILE" << 'EOF'
{
  "silentStartup": true,
  "showWelcome": false
}
EOF

clear

# Display SISO welcome screen
cat << 'EOF'
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
│  /help for help, /status for your current setup            │
│                                                             │
EOF

# Show current project info
CURRENT_DIR=$(basename "$PWD")
PROJECT_TYPE="Development"

# Detect project type
if [[ -f "package.json" ]]; then
    if grep -q "react" package.json 2>/dev/null; then
        PROJECT_TYPE="React/TypeScript"
    elif grep -q "next" package.json 2>/dev/null; then
        PROJECT_TYPE="Next.js"
    else
        PROJECT_TYPE="Node.js"
    fi
elif [[ -f "requirements.txt" ]] || [[ -f "pyproject.toml" ]]; then
    PROJECT_TYPE="Python"
elif [[ -f "Cargo.toml" ]]; then
    PROJECT_TYPE="Rust"
elif [[ -f "go.mod" ]]; then
    PROJECT_TYPE="Go"
fi

# Special detection for SISO projects
if [[ "$CURRENT_DIR" == *"SISO"* ]] || [[ "$PWD" == *"SISO"* ]]; then
    PROJECT_TYPE="🌟 SISO $PROJECT_TYPE"
fi

# Show project info
printf "│  🎯 Project: %-44s │\n" "$CURRENT_DIR"
printf "│  🔧 Type: %-47s │\n" "$PROJECT_TYPE"
printf "│  📍 cwd: %-48s │\n" "$PWD"

cat << 'EOF'
│                                                             │
│  💡 Ready to revolutionize your development workflow!       │
│                                                             │
╰─────────────────────────────────────────────────────────────╯

🔥 SISO Enhanced Claude Code is now starting...

EOF

# Give user a moment to see the welcome screen
read -p "Press Enter to continue to Claude Code..." -r

# Clear screen to keep only SISO welcome visible
clear

# Redisplay the SISO welcome screen one more time
cat << 'EOF'
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
│  🌟 SISO Enhanced Claude Code Session Active               │
│                                                             │
╰─────────────────────────────────────────────────────────────╯

EOF

# Launch Claude Code with all arguments passed through
if [ $# -eq 0 ]; then
    # No arguments provided - start interactive session
    exec claude --continue 2>/dev/null || exec claude
else
    # Arguments provided - pass them through
    exec claude "$@"
fi