#!/bin/bash

# Cross-Platform Notifier - Inspired by jamez01/claude-notify
MESSAGE="$1"
TITLE="${2:-SISO Claude Code}"
URGENCY="${3:-normal}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*) echo "linux" ;;
        CYGWIN*|MINGW*|MSYS*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

# macOS notifications
notify_macos() {
    local msg="$1"
    local title="$2"
    local urgency="$3"
    
    # Use osascript for native notifications
    osascript -e "display notification \"$msg\" with title \"$title\" sound name \"Glass\"" 2>/dev/null || {
        # Fallback to terminal bell
        echo -e "\a$title: $msg"
    }
}

# Linux notifications
notify_linux() {
    local msg="$1"
    local title="$2"
    local urgency="$3"
    
    if command -v notify-send >/dev/null; then
        notify-send --urgency="$urgency" "$title" "$msg" 2>/dev/null
    elif command -v zenity >/dev/null; then
        zenity --info --title="$title" --text="$msg" 2>/dev/null &
    else
        echo -e "\a$title: $msg"
    fi
}

# Windows notifications  
notify_windows() {
    local msg="$1"
    local title="$2"
    
    # PowerShell toast notification
    powershell.exe -Command "
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show('$msg', '$title', 'OK', 'Information')
    " 2>/dev/null || echo -e "\a$title: $msg"
}

# Smart notification content based on context
enhance_message() {
    local original="$1"
    
    # Add project context if available
    if [[ -f "package.json" ]]; then
        local project=$(grep '"name"' package.json | cut -d'"' -f4 2>/dev/null || echo "Unknown")
        echo "[$project] $original"
    else
        echo "$original"
    fi
}

# Notification throttling (prevent spam)
check_throttle() {
    local throttle_file="~/.claude/notification-throttle"
    local current_time=$(date +%s)
    
    if [[ -f "$throttle_file" ]]; then
        local last_notification=$(cat "$throttle_file")
        local time_diff=$((current_time - last_notification))
        
        # Throttle notifications to max 1 per 5 seconds
        if [[ $time_diff -lt 5 ]]; then
            return 1  # Skip notification
        fi
    fi
    
    echo "$current_time" > "$throttle_file"
    return 0  # Allow notification
}

# Log notification for debugging
log_notification() {
    echo "[$TIMESTAMP] NOTIFICATION: $1" >> ~/.claude/logs/notifications.log
}

# Main notification logic
main() {
    # Check throttling
    if ! check_throttle; then
        log_notification "THROTTLED: $MESSAGE"
        return 0
    fi
    
    # Enhance message with context
    local enhanced_msg=$(enhance_message "$MESSAGE")
    
    # Detect OS and send appropriate notification
    local os=$(detect_os)
    
    case "$os" in
        "macos")
            notify_macos "$enhanced_msg" "$TITLE" "$URGENCY"
            ;;
        "linux")
            notify_linux "$enhanced_msg" "$TITLE" "$URGENCY"
            ;;
        "windows")
            notify_windows "$enhanced_msg" "$TITLE"
            ;;
        *)
            echo -e "\a$TITLE: $enhanced_msg"
            ;;
    esac
    
    log_notification "$enhanced_msg"
}

# Initialize
mkdir -p ~/.claude/logs

# Execute
main