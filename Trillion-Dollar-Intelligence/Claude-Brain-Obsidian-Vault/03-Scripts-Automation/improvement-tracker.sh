#!/bin/bash

# Improvement Tracker - Log and analyze hook effectiveness
EVENT_TYPE="$1"  # hook_triggered, task_completed, error_prevented, suggestion_made
HOOK_NAME="$2"
DETAILS="$3"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DATE=$(date '+%Y-%m-%d')

# Create improvement tracking log
log_improvement() {
    local event="$1"
    local hook="$2"
    local details="$3"
    
    # Create structured log entry
    echo "{
        \"timestamp\": \"$TIMESTAMP\",
        \"date\": \"$DATE\",
        \"event_type\": \"$event\",
        \"hook_name\": \"$hook\",
        \"details\": \"$details\",
        \"project\": \"$(basename "$(pwd)")\",
        \"git_branch\": \"$(git branch --show-current 2>/dev/null || echo 'unknown')\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }" >> ~/.claude/analytics/improvements.jsonl
}

# Track specific improvement metrics
track_metrics() {
    local metric_type="$1"
    local value="$2"
    
    case "$metric_type" in
        "prevented_error")
            echo "$TIMESTAMP,prevented_error,$value" >> ~/.claude/analytics/metrics.csv
            ;;
        "auto_fixed")
            echo "$TIMESTAMP,auto_fixed,$value" >> ~/.claude/analytics/metrics.csv
            ;;
        "time_saved")
            echo "$TIMESTAMP,time_saved,$value" >> ~/.claude/analytics/metrics.csv
            ;;
        "suggestion_followed")
            echo "$TIMESTAMP,suggestion_followed,$value" >> ~/.claude/analytics/metrics.csv
            ;;
    esac
}

# Generate improvement insights
generate_insights() {
    echo "🔍 IMPROVEMENT INSIGHTS - $(date '+%Y-%m-%d')"
    echo "=============================================="
    
    # Hook effectiveness
    echo ""
    echo "📊 Hook Effectiveness Today:"
    if [[ -f ~/.claude/analytics/improvements.jsonl ]]; then
        grep "\"date\": \"$DATE\"" ~/.claude/analytics/improvements.jsonl | \
        jq -r '.hook_name' | sort | uniq -c | sort -nr | head -10
    else
        echo "No improvement data yet"
    fi
    
    # Error prevention
    echo ""
    echo "🛡️ Errors Prevented Today:"
    grep "$DATE,prevented_error" ~/.claude/analytics/metrics.csv 2>/dev/null | wc -l | \
    xargs echo "Total errors prevented:"
    
    # Time saved
    echo ""
    echo "⏰ Estimated Time Saved:"
    if [[ -f ~/.claude/analytics/metrics.csv ]]; then
        grep "$DATE,time_saved" ~/.claude/analytics/metrics.csv | \
        cut -d',' -f3 | awk '{sum+=$1} END {print "Total minutes saved:", sum}'
    else
        echo "No time tracking data yet"
    fi
    
    # Most active hooks
    echo ""
    echo "🔥 Most Active Hooks:"
    if [[ -f ~/.claude/analytics/improvements.jsonl ]]; then
        grep "\"date\": \"$DATE\"" ~/.claude/analytics/improvements.jsonl | \
        jq -r '.hook_name' | head -5
    fi
}

# Self-analysis of patterns
self_analyze() {
    echo ""
    echo "🧠 SELF-ANALYSIS INSIGHTS:"
    echo "========================="
    
    # Analyze patterns from the last 7 days
    local week_ago=$(date -d '7 days ago' '+%Y-%m-%d' 2>/dev/null || date -v-7d '+%Y-%m-%d' 2>/dev/null)
    
    if [[ -f ~/.claude/analytics/improvements.jsonl ]]; then
        # Most common issues prevented
        echo ""
        echo "📈 Top Issues Prevented (Last 7 Days):"
        grep -A1 -B1 "prevented_error" ~/.claude/analytics/improvements.jsonl | \
        jq -r '.details' 2>/dev/null | sort | uniq -c | sort -nr | head -5
        
        # Project with most improvements
        echo ""
        echo "🏆 Most Improved Project:"
        jq -r '.project' ~/.claude/analytics/improvements.jsonl 2>/dev/null | \
        sort | uniq -c | sort -nr | head -3
        
        # Hook success patterns
        echo ""
        echo "✅ Most Successful Hook Categories:"
        jq -r '.hook_name' ~/.claude/analytics/improvements.jsonl 2>/dev/null | \
        cut -d'-' -f1 | sort | uniq -c | sort -nr | head -5
    fi
    
    # Improvement trends
    echo ""
    echo "📊 Weekly Improvement Trend:"
    for i in {6..0}; do
        local check_date=$(date -d "$i days ago" '+%Y-%m-%d' 2>/dev/null || date -v-${i}d '+%Y-%m-%d' 2>/dev/null)
        local count=$(grep "\"date\": \"$check_date\"" ~/.claude/analytics/improvements.jsonl 2>/dev/null | wc -l)
        echo "$check_date: $count improvements"
    done
}

# Initialize directories
mkdir -p ~/.claude/analytics

# Main execution
case "$EVENT_TYPE" in
    "hook_triggered")
        log_improvement "$EVENT_TYPE" "$HOOK_NAME" "$DETAILS"
        ;;
    "error_prevented")
        log_improvement "$EVENT_TYPE" "$HOOK_NAME" "$DETAILS"
        track_metrics "prevented_error" "$DETAILS"
        ;;
    "auto_fixed")
        log_improvement "$EVENT_TYPE" "$HOOK_NAME" "$DETAILS"
        track_metrics "auto_fixed" "$DETAILS"
        ;;
    "time_saved")
        track_metrics "time_saved" "$HOOK_NAME"  # Hook name contains minutes
        ;;
    "generate_report")
        generate_insights
        self_analyze
        ;;
    *)
        log_improvement "$EVENT_TYPE" "$HOOK_NAME" "$DETAILS"
        ;;
esac