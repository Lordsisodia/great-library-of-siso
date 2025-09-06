#!/bin/bash

# Claude Brain Config Sync Monitor
# Monitors sync status and provides intelligent sync management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAIN_CONFIG_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$BRAIN_CONFIG_DIR/logs/sync-monitor.log"

cd "$BRAIN_CONFIG_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check GitHub status
check_github_status() {
    log "🔍 Checking GitHub repository status..."
    
    # Check if we can reach GitHub
    if ! curl -s --max-time 10 https://api.github.com/status > /dev/null; then
        log "⚠️ GitHub API unreachable"
        return 1
    fi
    
    # Check repository status
    local repo_status=$(curl -s "https://api.github.com/repos/Lordsisodia/claude-brain-config")
    if echo "$repo_status" | grep -q '"private": false'; then
        log "✅ GitHub repository accessible"
        return 0
    else
        log "❌ GitHub repository not accessible"
        return 1
    fi
}

# Function to check sync conflicts
check_conflicts() {
    log "🔍 Checking for sync conflicts..."
    
    git fetch origin
    
    local local_commit=$(git rev-parse HEAD)
    local remote_commit=$(git rev-parse origin/main)
    local base_commit=$(git merge-base HEAD origin/main)
    
    if [[ "$local_commit" == "$remote_commit" ]]; then
        log "✅ Local and remote are in sync"
        echo "SYNC_STATUS=in_sync"
    elif [[ "$local_commit" == "$base_commit" ]]; then
        log "📥 Remote has new commits (fast-forward possible)"
        echo "SYNC_STATUS=behind"
    elif [[ "$remote_commit" == "$base_commit" ]]; then
        log "📤 Local has new commits (push needed)"
        echo "SYNC_STATUS=ahead"
    else
        log "⚠️ Branches have diverged (merge required)"
        echo "SYNC_STATUS=diverged"
    fi
}

# Function to intelligent sync resolution
intelligent_sync() {
    local sync_status="$1"
    
    case "$sync_status" in
        "in_sync")
            log "✅ No sync needed"
            ;;
        "behind")
            log "📥 Pulling remote changes..."
            git pull origin main --no-edit
            log "✅ Successfully pulled remote changes"
            ;;
        "ahead")
            log "📤 Pushing local changes..."
            "$SCRIPT_DIR/auto-sync-local.sh"
            log "✅ Successfully pushed local changes"
            ;;
        "diverged")
            log "🔄 Resolving diverged branches..."
            # Create backup branch
            local backup_branch="backup-$(date +%Y%m%d-%H%M%S)"
            git branch "$backup_branch"
            log "📋 Created backup branch: $backup_branch"
            
            # Try to merge
            if git pull origin main --no-edit; then
                log "✅ Successfully merged diverged branches"
                git push origin main
            else
                log "❌ Merge conflicts detected - manual intervention required"
                log "🔧 Backup branch created: $backup_branch"
                return 1
            fi
            ;;
    esac
}

# Function to monitor file changes
monitor_changes() {
    log "👀 Starting file change monitoring..."
    
    # Use fswatch if available (macOS), otherwise fall back to basic monitoring
    if command -v fswatch > /dev/null; then
        log "📡 Using fswatch for real-time monitoring"
        fswatch -o "$BRAIN_CONFIG_DIR" --exclude=".git" --exclude="logs" | while read num; do
            log "📝 File changes detected ($num files)"
            sleep 5  # Debounce rapid changes
            "$SCRIPT_DIR/auto-sync-local.sh"
        done
    else
        log "⏰ Using periodic monitoring (every 5 minutes)"
        while true; do
            if [[ -n $(git status --porcelain) ]]; then
                log "📝 Changes detected during periodic check"
                "$SCRIPT_DIR/auto-sync-local.sh"
            fi
            sleep 300  # Check every 5 minutes
        done
    fi
}

# Function to setup intelligent sync daemon
setup_daemon() {
    log "🚀 Setting up Claude Brain Config Sync Daemon..."
    
    # Create daemon script
    cat > "$SCRIPT_DIR/sync-daemon.sh" << 'EOF'
#!/bin/bash
# Claude Brain Config Sync Daemon

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/brain-sync-monitor.sh"

# Start monitor in background
nohup "$MONITOR_SCRIPT" --monitor > /dev/null 2>&1 &
echo $! > "$SCRIPT_DIR/../logs/sync-daemon.pid"

echo "🧠 Claude Brain Config Sync Daemon started"
echo "PID: $(cat "$SCRIPT_DIR/../logs/sync-daemon.pid")"
EOF

    chmod +x "$SCRIPT_DIR/sync-daemon.sh"
    log "✅ Sync daemon created"
}

# Function to show sync status dashboard
show_dashboard() {
    clear
    echo "🧠 Claude Brain Config Sync Dashboard"
    echo "====================================="
    echo ""
    echo "Repository: https://github.com/Lordsisodia/claude-brain-config"
    echo "Directory: $BRAIN_CONFIG_DIR"
    echo "Time: $(date)"
    echo ""
    
    # Git status
    echo "📊 Git Status:"
    echo "---------------"
    git status --short || echo "No changes"
    echo ""
    
    # Sync status
    echo "🔄 Sync Status:"
    echo "---------------"
    check_conflicts
    echo ""
    
    # Recent commits
    echo "📝 Recent Commits:"
    echo "------------------"
    git log --oneline -5
    echo ""
    
    # Log tail
    echo "📋 Recent Log Entries:"
    echo "----------------------"
    if [[ -f "$LOG_FILE" ]]; then
        tail -5 "$LOG_FILE"
    else
        echo "No log entries yet"
    fi
}

# Main execution
main() {
    mkdir -p "$BRAIN_CONFIG_DIR/logs"
    
    case "$1" in
        --monitor)
            monitor_changes
            ;;
        --check)
            check_github_status && check_conflicts
            ;;
        --sync)
            local status=$(check_conflicts | grep "SYNC_STATUS=" | cut -d= -f2)
            intelligent_sync "$status"
            ;;
        --daemon)
            setup_daemon
            ;;
        --dashboard)
            show_dashboard
            ;;
        --help)
            echo "Claude Brain Config Sync Monitor"
            echo ""
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --monitor     Start real-time file monitoring"
            echo "  --check       Check sync status"
            echo "  --sync        Perform intelligent sync"
            echo "  --daemon      Setup sync daemon"
            echo "  --dashboard   Show sync dashboard"
            echo "  --help        Display this help message"
            ;;
        *)
            show_dashboard
            ;;
    esac
}

main "$@"