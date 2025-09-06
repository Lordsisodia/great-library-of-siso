#!/bin/bash

echo "📊 RATE-LIMITED AUTONOMOUS SESSION PROGRESS"
echo "============================================"

# Check if session is running
if [ -f "rate_limited_sessions/logs/current_session.pid" ]; then
    PID=$(cat rate_limited_sessions/logs/current_session.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Session RUNNING (PID: $PID)"
        
        # Get runtime
        RUNTIME=$(ps -o etime= -p $PID | xargs)
        echo "⏰ Runtime: $RUNTIME"
        
        # Find active session directory
        LATEST_SESSION=$(find rate_limited_sessions -maxdepth 1 -type d -name "202*" | sort | tail -1)
        if [ -d "$LATEST_SESSION" ]; then
            echo "📁 Active session: $(basename $LATEST_SESSION)"
            
            # Count files by type
            RESEARCH_COUNT=$(find "$LATEST_SESSION/research" -name "*.md" 2>/dev/null | wc -l | xargs)
            CODE_COUNT=$(find "$LATEST_SESSION/code" -name "*.py" 2>/dev/null | wc -l | xargs)
            ANALYSIS_COUNT=$(find "$LATEST_SESSION/analysis" -name "*.md" 2>/dev/null | wc -l | xargs)
            TOTAL_FILES=$((RESEARCH_COUNT + CODE_COUNT + ANALYSIS_COUNT))
            
            echo "📋 Content generated:"
            echo "  🔍 Research files: $RESEARCH_COUNT"
            echo "  💻 Code files: $CODE_COUNT" 
            echo "  🧠 Analysis files: $ANALYSIS_COUNT"
            echo "  📊 Total files: $TOTAL_FILES"
            
            # Calculate character count
            if [ "$TOTAL_FILES" -gt 0 ]; then
                TOTAL_CHARS=$(find "$LATEST_SESSION" -type f \( -name "*.py" -o -name "*.md" \) -exec wc -c {} + 2>/dev/null | tail -1 | awk '{print $1}')
                if [ -n "$TOTAL_CHARS" ] && [ "$TOTAL_CHARS" -gt 0 ]; then
                    echo "  📝 Total characters: $(printf "%'d" $TOTAL_CHARS)"
                    
                    # Calculate average file size
                    AVG_SIZE=$((TOTAL_CHARS / TOTAL_FILES))
                    echo "  📄 Average file size: $(printf "%'d" $AVG_SIZE) chars"
                    
                    # Estimate rate based on runtime (convert etime to seconds)
                    RUNTIME_SECONDS=$(echo $RUNTIME | awk -F: '{
                        if (NF==3) print $1*3600+$2*60+$3; 
                        else if (NF==2) print $1*60+$2; 
                        else print $1
                    }')
                    
                    if [ "$RUNTIME_SECONDS" -gt 0 ]; then
                        CHARS_PER_HOUR=$((TOTAL_CHARS * 3600 / RUNTIME_SECONDS))
                        echo "  ⚡ Generation rate: $(printf "%'d" $CHARS_PER_HOUR) chars/hour"
                        
                        # Project to 12 hours
                        PROJECTED_12H=$((CHARS_PER_HOUR * 12))
                        echo "  🎯 12-hour projection: $(printf "%'d" $PROJECTED_12H) characters"
                        
                        if [ "$PROJECTED_12H" -gt 1000000 ]; then
                            echo "  ✅ ON TRACK for 1M+ character target!"
                        else
                            echo "  ⚠️  May need optimization"
                        fi
                    fi
                fi
            fi
            
            # Check for hourly reports
            REPORTS_COUNT=$(find "$LATEST_SESSION/hourly_reports" -name "*.json" 2>/dev/null | wc -l | xargs)
            if [ "$REPORTS_COUNT" -gt 0 ]; then
                echo "  📈 Hourly reports: $REPORTS_COUNT"
                
                # Show latest report
                LATEST_REPORT=$(find "$LATEST_SESSION/hourly_reports" -name "*.json" | sort | tail -1)
                if [ -f "$LATEST_REPORT" ]; then
                    echo "  📊 Latest report: $(basename $LATEST_REPORT)"
                fi
            fi
            
            echo ""
            echo "📋 Latest files:"
            find "$LATEST_SESSION" -type f \( -name "*.py" -o -name "*.md" \) -exec ls -la {} + 2>/dev/null | tail -3
        fi
        
        echo ""
        echo "🔍 Live monitoring commands:"
        echo "  📜 tail -f rate_limited_sessions/logs/session_*.log"
        echo "  📊 watch -n 30 ./check-rate-limited-progress.sh"
        echo "  🛑 kill $PID"
        
    else
        echo "❌ Session NOT RUNNING (PID $PID not found)"
        echo "🔧 Restart with: ./launch-rate-limited-session.sh"
    fi
else
    echo "❌ No active session found"
    echo "🚀 Launch with: ./launch-rate-limited-session.sh"
fi

echo ""
echo "⏰ Expected completion: 12 hours from session start"
echo "🎯 Expected output: 1-3 million characters, 500+ files"