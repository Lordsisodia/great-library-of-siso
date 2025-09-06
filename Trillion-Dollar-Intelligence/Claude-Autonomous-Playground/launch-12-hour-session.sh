#!/bin/bash

echo "🚀 LAUNCHING 12-HOUR CONTINUOUS AUTONOMOUS SESSION"
echo "================================================="
echo "Target: 5-10 MILLION characters overnight"
echo "Duration: 12 hours of continuous AI generation"
echo "Models: Cerebras + Gemini dual orchestration"
echo ""

# Create logs directory
mkdir -p continuous_sessions/logs

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="continuous_sessions/logs/session_${TIMESTAMP}.log"

echo "📁 Logs: $LOG_FILE"
echo "⏰ Start: $(date)"
echo "⏰ Expected End: $(date -v+12H)"
echo ""

# Launch the continuous system in background with full logging
nohup python3 continuous-autonomous-system.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "🤖 Autonomous system launched!"
echo "📊 Process ID: $PID"
echo "📋 Log file: $LOG_FILE" 
echo ""
echo "✅ The system will now run for 12 hours generating millions of characters"
echo "✅ Check progress with: tail -f $LOG_FILE"
echo "✅ Stop early with: kill $PID"
echo ""
echo "💤 Safe to close terminal - system runs independently"
echo "📈 Expected output: 5-10 million characters, 200+ files"

# Save PID for easy stopping
echo "$PID" > continuous_sessions/logs/current_session.pid
echo "Session PID $PID saved to continuous_sessions/logs/current_session.pid"

echo ""
echo "🎯 AUTONOMOUS SESSION LAUNCHED - CHECK BACK IN 12 HOURS!"