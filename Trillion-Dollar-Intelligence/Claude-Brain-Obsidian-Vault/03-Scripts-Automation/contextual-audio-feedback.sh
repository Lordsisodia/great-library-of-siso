#!/bin/bash

# Contextual Audio Feedback - Inspired by ctoth/claudio
TOOL_NAME="$1"
EVENT_TYPE="$2"  # thinking, success, error, interaction
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Audio configuration
VOLUME="${CLAUDE_AUDIO_VOLUME:-0.5}"
SOUND_PACK="${CLAUDE_SOUND_PACK:-default}"
AUDIO_ENABLED="${CLAUDE_AUDIO_ENABLED:-true}"

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/audio.log; }

# Check if audio is enabled
if [[ "$AUDIO_ENABLED" != "true" ]]; then
    exit 0
fi

# Sound mappings based on tool and event
get_sound_file() {
    local tool="$1"
    local event="$2"
    
    case "$tool:$event" in
        # Git operations
        "*git*commit*:success") echo "victory.wav" ;;
        "*git*push*:success") echo "achievement.wav" ;;
        "*git*:error") echo "sad-trombone.wav" ;;
        
        # Build operations
        "*npm*build*:thinking") echo "processing.wav" ;;
        "*npm*build*:success") echo "build-complete.wav" ;;
        "*npm*build*:error") echo "build-failed.wav" ;;
        
        # Test operations
        "*test*:success") echo "test-pass.wav" ;;
        "*test*:error") echo "test-fail.wav" ;;
        
        # File operations
        "Edit:success"|"Write:success") echo "edit-complete.wav" ;;
        "MultiEdit:success") echo "multi-edit-complete.wav" ;;
        
        # Default sounds
        "*:thinking") echo "thinking.wav" ;;
        "*:success") echo "success.wav" ;;
        "*:error") echo "error.wav" ;;
        "*:interaction") echo "notification.wav" ;;
        
        *) echo "default.wav" ;;
    esac
}

# Play sound with fallback system
play_sound() {
    local sound_file="$1"
    local sound_path="~/.claude/sounds/$SOUND_PACK/$sound_file"
    
    # Multi-level fallback system
    if [[ -f "$sound_path" ]]; then
        # Premium: Custom sound files
        play_audio_file "$sound_path"
    elif command -v say >/dev/null; then
        # Good: Text-to-speech (macOS)
        play_tts_feedback "$TOOL_NAME" "$EVENT_TYPE"
    elif command -v espeak >/dev/null; then
        # Good: Text-to-speech (Linux)
        espeak "Tool $TOOL_NAME $EVENT_TYPE" 2>/dev/null &
    else
        # Basic: System beep
        play_system_beep "$EVENT_TYPE"
    fi
}

# Play audio file
play_audio_file() {
    local file="$1"
    
    if command -v afplay >/dev/null; then
        # macOS
        afplay "$file" --volume "$VOLUME" 2>/dev/null &
    elif command -v aplay >/dev/null; then
        # Linux ALSA
        aplay "$file" 2>/dev/null &
    elif command -v paplay >/dev/null; then
        # Linux PulseAudio
        paplay "$file" 2>/dev/null &
    else
        log "⚠️ No audio player available for file: $file"
    fi
}

# Text-to-speech feedback
play_tts_feedback() {
    local tool="$1"
    local event="$2"
    
    case "$event" in
        "success")
            say "Task completed successfully" --rate 200 2>/dev/null &
            ;;
        "error")
            say "Error occurred" --rate 180 2>/dev/null &
            ;;
        "thinking")
            say "Processing" --rate 220 2>/dev/null &
            ;;
        *)
            say "$tool $event" --rate 200 2>/dev/null &
            ;;
    esac
}

# System beep patterns
play_system_beep() {
    local event="$1"
    
    case "$event" in
        "success")
            # Two short beeps
            (osascript -e 'beep 1' && sleep 0.1 && osascript -e 'beep 1') 2>/dev/null &
            ;;
        "error")
            # One long beep
            osascript -e 'beep 3' 2>/dev/null &
            ;;
        "thinking")
            # Single soft beep
            osascript -e 'beep 1' 2>/dev/null &
            ;;
        *)
            osascript -e 'beep 1' 2>/dev/null &
            ;;
    esac
}

# Contextual message generation
generate_audio_message() {
    local tool="$1"
    local event="$2"
    
    case "$tool" in
        *"git"*"commit"*)
            echo "Git commit $event"
            ;;
        *"npm"*"build"*)
            echo "Build $event"
            ;;
        *"test"*)
            echo "Tests $event"
            ;;
        *)
            echo "$tool $event"
            ;;
    esac
}

# Initialize sound directories
init_audio_system() {
    mkdir -p ~/.claude/sounds/default
    mkdir -p ~/.claude/logs
    
    # Create default sound files if they don't exist
    # (In real implementation, these would be actual audio files)
    touch ~/.claude/sounds/default/default.wav
    touch ~/.claude/sounds/default/success.wav
    touch ~/.claude/sounds/default/error.wav
}

# Main execution
main() {
    init_audio_system
    
    local sound_file=$(get_sound_file "$TOOL_NAME" "$EVENT_TYPE")
    log "🔊 Playing audio feedback: $TOOL_NAME:$EVENT_TYPE → $sound_file"
    
    play_sound "$sound_file"
}

# Execute if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi