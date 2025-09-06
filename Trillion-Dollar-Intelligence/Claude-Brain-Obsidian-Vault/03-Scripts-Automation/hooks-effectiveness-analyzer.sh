#!/bin/bash

# Hooks Effectiveness Analyzer - Comprehensive analysis of hook performance
ANALYSIS_TYPE="${1:-daily}"  # daily, weekly, monthly, full

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DATE=$(date '+%Y-%m-%d')

echo "🔍 HOOKS EFFECTIVENESS ANALYSIS - $ANALYSIS_TYPE"
echo "=================================================="

# Initialize CSV header if doesn't exist
if [[ ! -f ~/.claude/analytics/metrics.csv ]]; then
    echo "timestamp,metric_type,value,details" > ~/.claude/analytics/metrics.csv
fi

# Analyze hook logs for effectiveness
analyze_hook_logs() {
    echo ""
    echo "📊 HOOK EXECUTION ANALYSIS:"
    echo "-------------------------"
    
    if [[ -f ~/.claude/logs/hooks.log ]]; then
        local total_hooks=$(wc -l < ~/.claude/logs/hooks.log)
        local successful_hooks=$(grep -c "✅\|completed\|success" ~/.claude/logs/hooks.log)
        local error_hooks=$(grep -c "❌\|error\|failed" ~/.claude/logs/hooks.log)
        
        echo "Total hook executions: $total_hooks"
        echo "Successful executions: $successful_hooks"
        echo "Failed executions: $error_hooks"
        
        if [[ $total_hooks -gt 0 ]]; then
            local success_rate=$(( (successful_hooks * 100) / total_hooks ))
            echo "Success rate: ${success_rate}%"
        fi
    else
        echo "No hook logs found yet"
    fi
}

# Analyze productivity metrics
analyze_productivity() {
    echo ""
    echo "⚡ PRODUCTIVITY IMPACT ANALYSIS:"
    echo "------------------------------"
    
    # Count automated tasks
    local auto_formats=$(grep -c "Auto-format\|prettier" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local auto_tests=$(grep -c "auto.*test\|test.*auto" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local auto_fixes=$(grep -c "eslint.*fix\|auto.*fix" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local prevented_errors=$(grep -c "BLOCKED\|prevented\|security" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    
    echo "🎨 Auto-formatting operations: $auto_formats"
    echo "🧪 Auto-test executions: $auto_tests"
    echo "🔧 Auto-fixes applied: $auto_fixes"
    echo "🛡️ Errors prevented: $prevented_errors"
    
    # Calculate time saved (rough estimates)
    local format_time_saved=$((auto_formats * 1))  # 1 min per format
    local test_time_saved=$((auto_tests * 2))      # 2 min per test run
    local fix_time_saved=$((auto_fixes * 3))       # 3 min per manual fix
    local error_time_saved=$((prevented_errors * 15)) # 15 min per prevented error
    
    local total_time_saved=$((format_time_saved + test_time_saved + fix_time_saved + error_time_saved))
    
    echo ""
    echo "⏰ ESTIMATED TIME SAVED:"
    echo "Formatting: ${format_time_saved} minutes"
    echo "Testing: ${test_time_saved} minutes"
    echo "Fixing: ${fix_time_saved} minutes"
    echo "Error prevention: ${error_time_saved} minutes"
    echo "TOTAL: ${total_time_saved} minutes (${total_time_saved} hours)"
    
    # Log the metrics
    echo "$TIMESTAMP,time_saved_total,$total_time_saved,daily_calculation" >> ~/.claude/analytics/metrics.csv
}

# Analyze security improvements
analyze_security() {
    echo ""
    echo "🛡️ SECURITY ENHANCEMENT ANALYSIS:"
    echo "--------------------------------"
    
    local blocked_commands=$(grep -c "BLOCKED\|dangerous\|security.*block" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local sensitive_warnings=$(grep -c "sensitive\|credentials\|secret" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local security_audits=$(grep -c "audit\|security.*scan" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    
    echo "🚫 Dangerous commands blocked: $blocked_commands"
    echo "⚠️ Sensitive data warnings: $sensitive_warnings"
    echo "🔍 Security audits performed: $security_audits"
    
    local security_score=$((blocked_commands * 10 + sensitive_warnings * 5 + security_audits * 3))
    echo "🏆 Security improvement score: $security_score"
    
    # Log security metrics
    echo "$TIMESTAMP,security_blocks,$blocked_commands,commands_prevented" >> ~/.claude/analytics/metrics.csv
    echo "$TIMESTAMP,security_score,$security_score,daily_calculation" >> ~/.claude/analytics/metrics.csv
}

# Analyze code quality improvements
analyze_quality() {
    echo ""
    echo "🎯 CODE QUALITY ANALYSIS:"
    echo "-----------------------"
    
    local tdd_enforcements=$(grep -c "TDD\|test.*first" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local documentation_generated=$(grep -c "documentation\|auto.*doc" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local style_corrections=$(grep -c "style\|format\|lint" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local complexity_warnings=$(grep -c "complexity\|refactor.*suggest" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    
    echo "🧪 TDD enforcements: $tdd_enforcements"
    echo "📚 Documentation generated: $documentation_generated"
    echo "🎨 Style corrections: $style_corrections"
    echo "⚡ Complexity warnings: $complexity_warnings"
    
    local quality_score=$((tdd_enforcements * 8 + documentation_generated * 5 + style_corrections * 2 + complexity_warnings * 3))
    echo "🏆 Quality improvement score: $quality_score"
    
    # Log quality metrics
    echo "$TIMESTAMP,quality_score,$quality_score,daily_calculation" >> ~/.claude/analytics/metrics.csv
}

# Analyze workflow efficiency
analyze_workflow() {
    echo ""
    echo "🔄 WORKFLOW EFFICIENCY ANALYSIS:"
    echo "------------------------------"
    
    if [[ -f ~/.claude/analytics/session-tracking.csv ]]; then
        local total_sessions=$(wc -l < ~/.claude/analytics/session-tracking.csv)
        local tools_per_session=$(awk -F',' '{print $3}' ~/.claude/analytics/session-tracking.csv | sort | uniq | wc -l)
        
        echo "Total sessions tracked: $total_sessions"
        echo "Average tools per session: $tools_per_session"
    fi
    
    # Most effective hooks
    echo ""
    echo "🔥 MOST EFFECTIVE HOOKS:"
    if [[ -f ~/.claude/logs/hooks.log ]]; then
        grep -o "hook.*:" ~/.claude/logs/hooks.log | sort | uniq -c | sort -nr | head -10
    fi
}

# Generate improvement recommendations
generate_recommendations() {
    echo ""
    echo "💡 IMPROVEMENT RECOMMENDATIONS:"
    echo "=============================="
    
    # Analyze patterns and suggest improvements
    local failed_hooks=$(grep -c "❌\|error\|failed" ~/.claude/logs/hooks.log 2>/dev/null || echo 0)
    local total_hooks=$(wc -l < ~/.claude/logs/hooks.log 2>/dev/null || echo 1)
    local failure_rate=$(( (failed_hooks * 100) / total_hooks ))
    
    if [[ $failure_rate -gt 10 ]]; then
        echo "⚠️ High hook failure rate ($failure_rate%) - Consider reviewing hook configurations"
    fi
    
    if [[ $auto_tests -lt 5 ]]; then
        echo "🧪 Low automated testing - Consider enabling more test automation hooks"
    fi
    
    if [[ $prevented_errors -eq 0 ]]; then
        echo "🛡️ No security blocks - Either very secure or hooks need tuning"
    fi
    
    echo ""
    echo "🎯 OPTIMIZATION SUGGESTIONS:"
    echo "- Monitor hooks with high failure rates"
    echo "- Increase automation for repetitive tasks"
    echo "- Fine-tune security sensitivity based on usage"
    echo "- Consider adding more workflow-specific hooks"
}

# Historical trend analysis
analyze_trends() {
    echo ""
    echo "📈 TREND ANALYSIS (Last 7 Days):"
    echo "==============================="
    
    if [[ -f ~/.claude/analytics/metrics.csv ]]; then
        echo "Daily time saved:"
        for i in {6..0}; do
            local check_date=$(date -d "$i days ago" '+%Y-%m-%d' 2>/dev/null || date -v-${i}d '+%Y-%m-%d' 2>/dev/null)
            local time_saved=$(grep "$check_date.*time_saved" ~/.claude/analytics/metrics.csv | tail -1 | cut -d',' -f3)
            echo "$check_date: ${time_saved:-0} minutes"
        done
    fi
}

# Main execution
analyze_hook_logs
analyze_productivity
analyze_security
analyze_quality
analyze_workflow
generate_recommendations

if [[ "$ANALYSIS_TYPE" == "weekly" ]] || [[ "$ANALYSIS_TYPE" == "full" ]]; then
    analyze_trends
fi

echo ""
echo "📊 Analysis completed at $TIMESTAMP"
echo "📁 Data stored in ~/.claude/analytics/"
echo "🔄 Run daily for trend tracking: ~/.claude/scripts/hooks-effectiveness-analyzer.sh"