#!/bin/bash
# Automatically create new specialized agents based on patterns

TASK_TYPE="$1"
BASE_AGENT="${2:-backend-developer}"
SPECIFIC_KNOWLEDGE="$3"
TOOLS="${4:-LS, Read, Write, Edit, MultiEdit, Bash, WebSearch}"

# Sanitize agent name
AGENT_NAME=$(echo "$TASK_TYPE-specialist" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
TIMESTAMP=$(date +%Y-%m-%d)
CREATION_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# Check if agent already exists
if [ -f "$HOME/.claude/agents/auto-$AGENT_NAME.md" ]; then
    echo "⚠️  Agent $AGENT_NAME already exists. Updating instead..."
    # Append to existing agent's learnings
    echo "

## Auto-Learning Update - $TIMESTAMP
$SPECIFIC_KNOWLEDGE" >> "$HOME/.claude/agents/auto-$AGENT_NAME.md"
    exit 0
fi

# Create new agent
cat > "$HOME/.claude/agents/auto-$AGENT_NAME.md" << EOF
---
name: $AGENT_NAME
description: AUTO-GENERATED specialist for $TASK_TYPE tasks. Created $TIMESTAMP based on repeated patterns. This agent self-improves through usage.
tools: $TOOLS
parent: $BASE_AGENT
auto_generated: true
performance_threshold: 0.85
creation_time: "$CREATION_TIME"
---

# $AGENT_NAME - Self-Improving Specialist

## Mission
Specialized agent for **$TASK_TYPE** tasks, automatically created after detecting repeated patterns.
This agent learns and improves with each use.

## Core Competencies (Auto-Detected)
$SPECIFIC_KNOWLEDGE

## Learned Patterns
_This section updates automatically as I complete tasks_

## Workflow (Inherited from $BASE_AGENT)
1. **Analyze Request** - Understand specific $TASK_TYPE requirements
2. **Apply Specialized Knowledge** - Use patterns learned from previous tasks
3. **Execute with Precision** - Implement using optimized approaches
4. **Document Learnings** - Capture new patterns for future improvement

## Self-Improvement Log
- **Created**: $TIMESTAMP
- **Base Template**: $BASE_AGENT
- **Specialization**: $TASK_TYPE
- **Performance Target**: 85% success rate
- **Total Tasks**: 0
- **Success Rate**: N/A

## Known Solutions
_Automatically populated from successful task completions_

## Common Pitfalls
_Automatically populated from failed attempts_

## Integration Points
- Works with MCP tools when available
- Coordinates with other specialists via @agent-tech-lead-orchestrator
- Shares learnings with parent agent: $BASE_AGENT

---
*This agent self-improves. Each successful task adds to its knowledge base.*
EOF

echo "✨ Created new autonomous agent: $AGENT_NAME"
echo "📁 Location: ~/.claude/agents/auto-$AGENT_NAME.md"
echo "🚀 Usage: @agent-$AGENT_NAME"

# Log creation
mkdir -p ~/.claude/analytics
echo "{
  \"event\": \"agent_created\",
  \"agent\": \"$AGENT_NAME\",
  \"base\": \"$BASE_AGENT\",
  \"task_type\": \"$TASK_TYPE\",
  \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"
}" >> ~/.claude/analytics/agent-lifecycle.jsonl

# Create improvement tracker
cat > "$HOME/.claude/analytics/agent-$AGENT_NAME-improvements.md" << EOF
# Improvement Tracking: $AGENT_NAME

## Creation Context
- Date: $TIMESTAMP
- Reason: Repeated $TASK_TYPE tasks detected
- Base Agent: $BASE_AGENT

## Performance Metrics
| Date | Tasks | Success Rate | Avg Time | Notes |
|------|-------|--------------|----------|-------|
| $TIMESTAMP | 0 | N/A | N/A | Created |

## Learned Patterns
<!-- Auto-populated from successful tasks -->

## Failed Approaches
<!-- Auto-populated from unsuccessful attempts -->

## Evolution History
<!-- Track major updates and improvements -->
EOF

echo "📊 Tracking file created: ~/.claude/analytics/agent-$AGENT_NAME-improvements.md"