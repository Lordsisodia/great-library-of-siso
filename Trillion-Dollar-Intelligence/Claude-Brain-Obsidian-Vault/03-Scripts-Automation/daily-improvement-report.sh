#!/bin/bash

# Daily Improvement Report - Automated daily analysis
DATE=$(date '+%Y-%m-%d')
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

clear
echo "📊 CLAUDE CODE HOOKS - DAILY IMPROVEMENT REPORT"
echo "=============================================="
echo "Date: $DATE"
echo "Generated: $TIMESTAMP"
echo ""

# Quick statistics
echo "⚡ QUICK STATS:"
echo "-------------"

# Count today's activities
TODAYS_HOOKS=$(grep "$DATE" ~/.claude/logs/hooks.log 2>/dev/null | wc -l || echo 0)
TODAYS_BRIEFS=$(grep "$DATE" ~/.claude/logs/task-briefs.log 2>/dev/null | wc -l || echo 0)
ERRORS_PREVENTED=$(grep -c "BLOCKED\|prevented\|security" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)

echo "🔧 Hooks executed today: $TODAYS_HOOKS"
echo "📝 Tasks completed: $TODAYS_BRIEFS"
echo "🛡️ Errors prevented: $ERRORS_PREVENTED"

# Run comprehensive analysis
echo ""
~/.claude/scripts/hooks-effectiveness-analyzer.sh daily

# Show most active hooks today
echo ""
echo "🔥 MOST ACTIVE HOOKS TODAY:"
echo "-------------------------"
if [[ -f ~/.claude/logs/hooks.log ]]; then
    grep "$DATE" ~/.claude/logs/hooks.log | \
    grep -o "hook.*:" | sort | uniq -c | sort -nr | head -5
else
    echo "No hook activity logged yet"
fi

# Show task completion patterns
echo ""
echo "📋 TASK COMPLETION PATTERNS:"
echo "---------------------------"
if [[ -f ~/.claude/logs/task-briefs.log ]]; then
    grep "$DATE" ~/.claude/logs/task-briefs.log | \
    cut -d']' -f2 | cut -d':' -f2 | sort | uniq -c | sort -nr | head -5
else
    echo "No task briefs logged yet"
fi

# Project activity
echo ""
echo "🏢 PROJECT ACTIVITY:"
echo "------------------"
if [[ -f ~/.claude/analytics/session-tracking.csv ]]; then
    grep "$DATE" ~/.claude/analytics/session-tracking.csv | \
    cut -d',' -f4 | sort | uniq -c | sort -nr
else
    echo "No session tracking data yet"
fi

# Improvement suggestions
echo ""
echo "💡 TODAY'S INSIGHTS:"
echo "------------------"

# Analyze patterns and provide insights
if [[ $TODAYS_HOOKS -gt 50 ]]; then
    echo "🚀 High activity day! You've been very productive."
elif [[ $TODAYS_HOOKS -gt 20 ]]; then
    echo "✅ Good activity level. Hooks are working well."
elif [[ $TODAYS_HOOKS -lt 5 ]]; then
    echo "🤔 Low activity. Consider using Claude Code more actively."
fi

if [[ $ERRORS_PREVENTED -gt 0 ]]; then
    echo "🛡️ Great! Hooks prevented $ERRORS_PREVENTED potential issues today."
fi

# Check for patterns
if grep -q "TDD" ~/.claude/logs/hooks.log 2>/dev/null; then
    echo "🧪 TDD enforcement is active - quality development practices!"
fi

if grep -q "auto.*doc" ~/.claude/logs/hooks.log 2>/dev/null; then
    echo "📚 Documentation automation is working - knowledge preservation!"
fi

# Weekly trend
echo ""
echo "📈 WEEKLY TREND:"
echo "--------------"
for i in {6..0}; do
    local check_date=$(date -d "$i days ago" '+%Y-%m-%d' 2>/dev/null || date -v-${i}d '+%Y-%m-%d' 2>/dev/null)
    local daily_hooks=$(grep "$check_date" ~/.claude/logs/hooks.log 2>/dev/null | wc -l || echo 0)
    echo "$check_date: $daily_hooks hooks"
done

# Action items
echo ""
echo "🎯 RECOMMENDED ACTIONS:"
echo "====================="

# Analyze data and provide recommendations
if [[ -f ~/.claude/analytics/metrics.csv ]]; then
    local time_saved=$(grep "$DATE.*time_saved" ~/.claude/analytics/metrics.csv | tail -1 | cut -d',' -f3 2>/dev/null || echo 0)
    if [[ $time_saved -gt 60 ]]; then
        echo "⏰ Excellent! Saved $time_saved minutes today through automation."
    elif [[ $time_saved -gt 30 ]]; then
        echo "⏰ Good! Saved $time_saved minutes. Consider enabling more automation."
    else
        echo "⏰ Consider reviewing hook configuration for more time savings."
    fi
fi

echo "📊 Run full analysis: ~/.claude/scripts/hooks-effectiveness-analyzer.sh weekly"
echo "🔍 View logs: tail -f ~/.claude/logs/hooks.log"
echo "📈 Track metrics: cat ~/.claude/analytics/metrics.csv"

echo ""
echo "📅 Next report: $(date -d 'tomorrow' '+%Y-%m-%d' 2>/dev/null || date -v+1d '+%Y-%m-%d' 2>/dev/null)"
echo "🔄 Auto-run: Add to crontab or run manually daily"

# Save report
REPORT_FILE="~/.claude/reports/daily-$(date +%Y%m%d).txt"
mkdir -p ~/.claude/reports
{
    echo "CLAUDE CODE HOOKS - DAILY IMPROVEMENT REPORT"
    echo "Date: $DATE"
    echo "Hooks executed: $TODAYS_HOOKS"
    echo "Tasks completed: $TODAYS_BRIEFS"
    echo "Errors prevented: $ERRORS_PREVENTED"
    echo "Generated: $TIMESTAMP"
} > "$REPORT_FILE"

echo ""
echo "💾 Report saved: $REPORT_FILE"