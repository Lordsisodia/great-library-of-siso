#!/bin/bash

echo "📊 AUTONOMOUS SESSION PROGRESS CHECK"
echo "====================================="

# Check if session is running
if [ -f "continuous_sessions/logs/current_session.pid" ]; then
    PID=$(cat continuous_sessions/logs/current_session.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Session RUNNING (PID: $PID)"
        
        # Get session start time from process
        START_TIME=$(ps -o lstart= -p $PID | xargs)
        echo "⏰ Started: $START_TIME"
        
        # Count files and estimate characters
        LATEST_SESSION=$(find continuous_sessions -maxdepth 1 -type d -name "202*" | sort | tail -1)
        if [ -d "$LATEST_SESSION" ]; then
            FILE_COUNT=$(find "$LATEST_SESSION" -type f \( -name "*.py" -o -name "*.md" -o -name "*.json" \) | wc -l | xargs)
            echo "📁 Files created: $FILE_COUNT"
            
            # Estimate characters from file sizes
            if [ "$FILE_COUNT" -gt 0 ]; then
                TOTAL_SIZE=$(find "$LATEST_SESSION" -type f \( -name "*.py" -o -name "*.md" \) -exec wc -c {} + 2>/dev/null | tail -1 | awk '{print $1}' | xargs)
                if [ -n "$TOTAL_SIZE" ] && [ "$TOTAL_SIZE" -gt 0 ]; then
                    echo "📊 Characters generated: $(printf "%'d" $TOTAL_SIZE)"
                    
                    # Calculate rate
                    RUNTIME_SECONDS=$(ps -o etime= -p $PID | awk -F: '{if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2}' | xargs)
                    if [ "$RUNTIME_SECONDS" -gt 0 ]; then
                        CHARS_PER_HOUR=$((TOTAL_SIZE * 3600 / RUNTIME_SECONDS))
                        echo "⚡ Rate: $(printf "%'d" $CHARS_PER_HOUR) chars/hour"
                        
                        # Project to 12 hours
                        PROJECTED_TOTAL=$((CHARS_PER_HOUR * 12))
                        echo "🎯 12-hour projection: $(printf "%'d" $PROJECTED_TOTAL) characters"
                        
                        if [ "$PROJECTED_TOTAL" -gt 5000000 ]; then
                            echo "✅ ON TRACK to exceed 5M character target"
                        else
                            echo "⚠️  May fall short of 5M character target"
                        fi
                    fi
                fi
            fi
            
            # Show latest files
            echo ""
            echo "📋 Latest outputs:"
            find "$LATEST_SESSION" -type f \( -name "*.py" -o -name "*.md" \) -exec ls -la {} + 2>/dev/null | tail -5
        fi
        
        echo ""
        echo "🔍 Live log tail:"
        echo "tail -f continuous_sessions/logs/session_*.log"
        
    else
        echo "❌ Session NOT RUNNING (PID $PID not found)"
        echo "🔧 Restart with: ./launch-12-hour-session.sh"
    fi
else
    echo "❌ No active session found"
    echo "🚀 Launch with: ./launch-12-hour-session.sh"
fi

echo ""
echo "🛑 Stop session: kill \$(cat continuous_sessions/logs/current_session.pid)"