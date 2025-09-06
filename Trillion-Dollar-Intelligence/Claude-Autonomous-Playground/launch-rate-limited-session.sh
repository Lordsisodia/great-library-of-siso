#!/bin/bash

echo "🚀 LAUNCHING RATE-LIMITED 12-HOUR AUTONOMOUS SESSION"
echo "===================================================="
echo "📊 Respects API rate limits while maximizing output"
echo "🧠 Cerebras: 500 requests/hour (safe from 600 limit)"
echo "🤖 Gemini: 600 requests/hour (safe from 900 limit)" 
echo "🎯 Target: 1M+ characters through ~13,200 API calls"
echo ""

# Create logs directory
mkdir -p rate_limited_sessions/logs

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="rate_limited_sessions/logs/session_${TIMESTAMP}.log"

echo "📁 Logs: $LOG_FILE"
echo "⏰ Start: $(date)"
echo "⏰ Expected End: $(date -v+12H)"
echo ""

# Launch the rate-limited system in background
nohup python3 rate-limited-autonomous-system.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "🤖 Rate-limited autonomous system launched!"
echo "📊 Process ID: $PID"
echo "📋 Log file: $LOG_FILE"
echo ""
echo "✅ System respects all API rate limits"
echo "✅ Cerebras: Max 6,000 calls (safe from 14,400 daily limit)"
echo "✅ Gemini: Max 7,200 calls (safe from 10,800 daily limit)"
echo "✅ Expected output: 1-3 million characters, 500+ files"
echo ""
echo "🔍 Monitor progress: tail -f $LOG_FILE"
echo "🛑 Stop early: kill $PID"
echo ""

# Save PID for easy management
echo "$PID" > rate_limited_sessions/logs/current_session.pid
echo "Session PID saved for management"

echo ""
echo "🎯 RATE-COMPLIANT AUTONOMOUS SESSION LAUNCHED!"
echo "💤 Safe to sleep - no rate limit violations possible"