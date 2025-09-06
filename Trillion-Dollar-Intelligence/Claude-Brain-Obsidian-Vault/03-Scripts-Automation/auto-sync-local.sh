#!/bin/bash

# Claude Brain Config Auto-Sync Script
# Automatically syncs local changes with GitHub repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAIN_CONFIG_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BRAIN_CONFIG_DIR"

echo "🧠 Claude Brain Config Auto-Sync Starting..."
echo "Directory: $BRAIN_CONFIG_DIR"
echo "Time: $(date)"
echo "----------------------------------------"

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "CLAUDE-SYSTEM-ACTIVATOR.md" ]]; then
        echo "❌ Error: Not in claude-brain-config directory"
        exit 1
    fi
}

# Function to pull latest changes
pull_changes() {
    echo "📥 Pulling latest changes from GitHub..."
    git fetch origin
    
    if git diff HEAD origin/main --quiet; then
        echo "✅ Local is up to date with remote"
    else
        echo "🔄 Merging remote changes..."
        git merge origin/main --no-edit
    fi
}

# Function to check for local changes
check_local_changes() {
    if [[ -n $(git status --porcelain) ]]; then
        return 0  # Changes exist
    else
        return 1  # No changes
    fi
}

# Function to commit and push changes
push_changes() {
    echo "📤 Pushing local changes to GitHub..."
    
    # Add all changes
    git add .
    
    # Create detailed commit message
    local timestamp=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    local changed_files=$(git diff --cached --name-only | wc -l | tr -d ' ')
    
    git commit -m "🧠 Auto-sync: Brain config updates $timestamp

- $changed_files files updated
- Intelligence system enhancements
- Configuration optimizations
- Automated local sync

🤖 Generated with Claude Brain Auto-Sync
Co-Authored-By: Claude <noreply@anthropic.com>"
    
    # Push to remote
    git push origin main
    echo "✅ Successfully pushed to GitHub"
}

# Function to setup auto-sync cron job
setup_cron() {
    local cron_command="*/15 * * * * cd '$BRAIN_CONFIG_DIR' && '$SCRIPT_DIR/auto-sync-local.sh' >> '$BRAIN_CONFIG_DIR/logs/auto-sync.log' 2>&1"
    
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "auto-sync-local.sh"; then
        echo "⏰ Auto-sync cron job already exists"
    else
        echo "⏰ Setting up auto-sync cron job (every 15 minutes)..."
        (crontab -l 2>/dev/null; echo "$cron_command") | crontab -
        echo "✅ Auto-sync cron job added"
    fi
}

# Function to create log directory
setup_logging() {
    mkdir -p "$BRAIN_CONFIG_DIR/logs"
}

# Main execution
main() {
    check_directory
    setup_logging
    
    # Pull latest changes first
    pull_changes
    
    # Check for local changes and push if any
    if check_local_changes; then
        echo "📝 Local changes detected"
        push_changes
    else
        echo "✅ No local changes to sync"
    fi
    
    # Setup cron job if --setup flag is provided
    if [[ "$1" == "--setup" ]]; then
        setup_cron
    fi
    
    echo "----------------------------------------"
    echo "🧠 Auto-sync completed successfully!"
    echo "Repository: https://github.com/Lordsisodia/claude-brain-config"
}

# Handle script arguments
case "$1" in
    --help)
        echo "Claude Brain Config Auto-Sync Script"
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --setup    Setup automatic cron job (runs every 15 minutes)"
        echo "  --help     Display this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Run sync once"
        echo "  $0 --setup      # Setup automatic syncing"
        ;;
    *)
        main "$@"
        ;;
esac