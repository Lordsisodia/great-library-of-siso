#!/bin/bash
# Track agent performance and suggest improvements

AGENT_NAME="$1"
TASK_OUTCOME="$2"  # success/failure
TASK_DURATION="$3"
LEARNINGS="$4"

# Create analytics directory if needed
mkdir -p ~/.claude/analytics

# Log performance data
echo "{
  \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
  \"agent\": \"$AGENT_NAME\",
  \"outcome\": \"$TASK_OUTCOME\",
  \"duration\": \"$TASK_DURATION\",
  \"learnings\": \"$LEARNINGS\"
}" >> ~/.claude/analytics/agent-performance.jsonl

# Calculate success rate (using awk for compatibility)
TOTAL=$(grep -c "\"agent\": \"$AGENT_NAME\"" ~/.claude/analytics/agent-performance.jsonl)
SUCCESS=$(grep "\"agent\": \"$AGENT_NAME\"" ~/.claude/analytics/agent-performance.jsonl | grep -c "\"outcome\": \"success\"")

if [ "$TOTAL" -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", $SUCCESS/$TOTAL}")
    
    echo "📊 Agent Performance: $AGENT_NAME"
    echo "   Total tasks: $TOTAL"
    echo "   Success rate: $SUCCESS_RATE"
    
    # Check if improvement needed
    if (( $(awk "BEGIN {print ($SUCCESS_RATE < 0.8)}") )); then
        echo "⚠️  Agent $AGENT_NAME success rate below 80%. Consider improvements:"
        echo "   - Review failed tasks for patterns"
        echo "   - Update agent knowledge base"
        echo "   - Consider creating specialized variant"
        
        # Analyze failure patterns
        echo ""
        echo "🔍 Recent failures:"
        grep "\"agent\": \"$AGENT_NAME\"" ~/.claude/analytics/agent-performance.jsonl | \
        grep "\"outcome\": \"failure\"" | \
        tail -3 | \
        jq -r '.learnings' 2>/dev/null || echo "   No failure details recorded"
    fi
    
    # Suggest specialization if high usage
    if [ "$TOTAL" -gt 10 ]; then
        echo ""
        echo "💡 High usage detected. Consider creating specialized variants for common patterns."
    fi
fi

# Track patterns for auto-generation
if [ "$TASK_OUTCOME" = "success" ] && [ -n "$LEARNINGS" ]; then
    echo "{
      \"agent\": \"$AGENT_NAME\",
      \"pattern\": \"$LEARNINGS\",
      \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"
    }" >> ~/.claude/analytics/successful-patterns.jsonl
fi