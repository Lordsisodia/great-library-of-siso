#!/bin/bash
# Add these to your ~/.zshrc or ~/.bashrc

# Quick launchers
alias claude="~/DEV/claude-launcher.sh"
alias claudia="cd ~/DEV/claudia-gui && bun run tauri dev"
alias claude-web="cd ~/DEV/claudecodeui && npm run dev"
alias claude-mobile="echo 'Mobile URL: http://$(ipconfig getifaddr en0):3001'"

# Status check
alias claude-status="~/DEV/claude-launcher.sh 5"

# Stop all
alias claude-stop="pkill -f 'tauri dev'; pkill -f 'npm run dev'"

echo "Claude aliases configured!"
echo ""
echo "Available commands:"
echo "  claude       - Launch menu"
echo "  claudia      - Desktop app only"
echo "  claude-web   - Web interface only"
echo "  claude-mobile - Get mobile URL"
echo "  claude-status - Check what's running"
echo "  claude-stop  - Stop everything"