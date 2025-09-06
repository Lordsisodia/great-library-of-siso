#!/bin/bash

# 10x Hooks Dashboard - Real-time monitoring and insights
clear

echo "🚀 SISO Claude Code Hooks Dashboard"
echo "=================================="
echo ""

# System status
echo "📊 SYSTEM STATUS"
echo "---------------"
echo "Hooks Config: $(test -f ~/.claude/settings.hooks.json && echo '✅ Active' || echo '❌ Missing')"
echo "Scripts Dir:  $(test -d ~/.claude/scripts && echo '✅ Ready' || echo '❌ Missing')"
echo "Logs Dir:     $(test -d ~/.claude/logs && echo '✅ Ready' || echo '❌ Missing')"
echo ""

# Recent activity
echo "⚡ RECENT ACTIVITY (Last 10 operations)"
echo "--------------------------------------"
if [[ -f ~/.claude/logs/hooks.log ]]; then
    tail -10 ~/.claude/logs/hooks.log | while read line; do
        echo "  $line"
    done
else
    echo "  No activity logged yet"
fi
echo ""

# Analytics summary
echo "📈 TODAY'S ANALYTICS"
echo "-------------------"
DATE=$(date '+%Y-%m-%d')

# Session metrics
if [[ -f ~/.claude/logs/session-$DATE.log ]]; then
    sessions=$(wc -l < ~/.claude/logs/session-$DATE.log)
    echo "Sessions: $sessions"
else
    echo "Sessions: 0"
fi

# File operations
if [[ -f ~/.claude/analytics/siso-activity.csv ]]; then
    today_edits=$(grep "$DATE" ~/.claude/analytics/siso-activity.csv | wc -l)
    echo "SISO Edits: $today_edits"
else
    echo "SISO Edits: 0"
fi

# Build metrics
if [[ -f ~/.claude/analytics/build-metrics.csv ]]; then
    builds_today=$(grep "$DATE" ~/.claude/analytics/build-metrics.csv | wc -l)
    echo "Builds: $builds_today"
else
    echo "Builds: 0"
fi

echo ""

# Quick actions
echo "🎮 QUICK ACTIONS"
echo "---------------"
echo "1. View live logs:     tail -f ~/.claude/logs/hooks.log"
echo "2. Check session:      cat ~/.claude/logs/session-$DATE.log"
echo "3. View analytics:     ls ~/.claude/analytics/"
echo "4. Reset hooks:        rm ~/.claude/logs/* && echo 'Logs cleared'"
echo ""

# Performance insights
echo "💡 PERFORMANCE INSIGHTS"
echo "----------------------"
if [[ -f ~/.claude/analytics/build-metrics.csv ]]; then
    avg_build_time=$(awk -F',' '{sum+=$2; count++} END {print int(sum/count)}' ~/.claude/analytics/build-metrics.csv 2>/dev/null || echo "0")
    echo "Average build time: ${avg_build_time}s"
else
    echo "No build metrics yet"
fi

if [[ -f ~/.claude/analytics/siso-activity.csv ]]; then
    total_siso_edits=$(wc -l < ~/.claude/analytics/siso-activity.csv)
    echo "Total SISO edits: $total_siso_edits"
else
    echo "No SISO activity yet"
fi

echo ""
echo "🔄 Dashboard refreshed at $(date '+%H:%M:%S')"