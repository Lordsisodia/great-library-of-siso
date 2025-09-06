#!/bin/bash

# Universal Claude Enhancement - Makes SISO the default experience
# Replaces claude command with intelligent SISO-enhanced version

# Store original Claude location
ORIGINAL_CLAUDE="/Users/shaansisodia/.npm-global/bin/claude-original"

# If this is first time setup, backup original Claude
if [[ ! -f "$ORIGINAL_CLAUDE" ]]; then
    echo "🔧 Setting up universal SISO enhancement..."
    
    # Find current Claude location
    CLAUDE_PATH=$(which claude)
    
    if [[ -z "$CLAUDE_PATH" ]]; then
        echo "❌ Claude not found in PATH"
        exit 1
    fi
    
    # Backup original Claude
    cp "$CLAUDE_PATH" "$ORIGINAL_CLAUDE"
    echo "✅ Original Claude backed up to: $ORIGINAL_CLAUDE"
fi

# Check for vanilla flag (escape hatch)
if [[ "$1" == "--vanilla" ]]; then
    shift
    exec "$ORIGINAL_CLAUDE" "$@"
fi

# Show SISO welcome with real-time intelligence
~/.claude/scripts/siso-dynamic-welcome.sh > /tmp/siso-context.json 2>/dev/null &
CONTEXT_PID=$!

# Display SISO welcome immediately (no delays)
cat << 'EOF'
╭─────────────────────────────────────────────────────────────╮
│   ███████╗██╗███████╗ ██████╗                              │
│   ██╔════╝██║██╔════╝██╔═══██╗                             │
│   ███████╗██║███████╗██║   ██║                             │
│   ╚════██║██║╚════██║██║   ██║                             │
│   ███████║██║███████║╚██████╔╝                             │
│   ╚══════╝╚═╝╚══════╝ ╚═════╝                              │
│                                                             │
│  🚀 SISO Enhanced Claude Code                               │
│  /help for help, /status for your current setup            │
│                                                             │
EOF

# Show current directory
printf "│  📍 cwd: %-48s │\n" "$PWD"

# Wait for context loading to complete (but don't block user)
wait $CONTEXT_PID 2>/dev/null

echo "╰─────────────────────────────────────────────────────────────╯"
echo ""

# Launch original Claude with all functionality intact
exec "$ORIGINAL_CLAUDE" "$@"