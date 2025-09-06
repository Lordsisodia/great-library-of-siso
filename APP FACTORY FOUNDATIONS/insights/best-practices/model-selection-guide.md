# AI Model Selection Guide for Different Tasks

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## Core Principle: Right Model for Right Task

**Goal**: Use the smallest AI model possible for each task
- **Smallest** = Cheapest + Fastest
- **Right-sized** = Not too weak, not too strong
- **Task-matched** = Model capabilities match task complexity

## Model Recommendations by Task

### GPT-4o-mini: Quick Edits
- **Use for**: Simple code changes, typos, small refactors
- **Trigger**: Comment + K in most editors
- **Speed**: Fastest
- **Cost**: Cheapest
- **Best for**: Single-line changes, obvious fixes

### GPT-5: Analysis & Planning
- **Use for**: 
  - PRD creation and scoping
  - Complex feature analysis
  - Code analysis tasks
  - Task breakdown and planning
- **Strength**: "Extremely good at analytics tasks"
- **Best for**: Strategic thinking, not implementation

### Claude Sonnet: General Coding
- **Use for**: Most coding workflows
- **Current status**: "My preferred coding model today"
- **Sweet spot**: Balance of capability and efficiency
- **Best for**: Feature implementation, debugging, refactoring

### Claude Opus: One-Shot Complex Features
- **Use for**: When you know you can complete feature in one go
- **Strategy**: Can be more cost-effective than Sonnet if it one-shots
- **Risk**: Higher cost if it fails and needs multiple attempts
- **Best for**: Well-defined, complex features with clear requirements

## Task-Specific Selection Strategy

### Planning Phase (Use GPT-5):
```
✅ Creating PRDs
✅ Breaking down complex requirements  
✅ Scoping tasks and dependencies
✅ Analyzing existing code architecture
✅ Asking clarifying questions
```

### Quick Fixes (Use GPT-4o-mini):
```
✅ Fixing typos
✅ Simple variable renaming
✅ Adding missing imports
✅ Basic formatting changes
✅ Single-line bug fixes
```

### Implementation (Use Sonnet):
```
✅ Building new features
✅ Refactoring existing code
✅ Writing tests
✅ Debugging complex issues
✅ General development work
```

### Complex One-Shots (Use Opus):
```
✅ Complete feature implementation (if confident)
✅ Complex architectural changes
✅ Large refactoring projects
✅ End-to-end integrations
```

## Cost-Effectiveness Analysis

### When Opus Beats Sonnet:
- **Scenario**: You're confident it can one-shot the feature
- **Math**: High upfront cost but completes in single attempt
- **Risk**: If it fails, becomes most expensive option

### When Sonnet is Best:
- **Scenario**: Most general development tasks
- **Math**: Good balance of capability and cost
- **Reliability**: Consistent performance across task types

### When GPT-5 is Essential:
- **Scenario**: Strategic planning and analysis
- **Value**: Prevents costly mistakes in implementation
- **ROI**: Small planning cost saves large implementation costs

## Multi-Model Workflow Strategy

### Recommended Flow:
1. **GPT-5**: Plan and scope the entire project
2. **GPT-5**: Break into specific tasks with dependencies
3. **Sonnet**: Implement most tasks
4. **GPT-4o-mini**: Quick fixes and adjustments
5. **Opus**: One-shot complex tasks (when confident)

### Parallel Agent Strategy:
```
Agent 1 (Sonnet): Backend API implementation
Agent 2 (Sonnet): Frontend components  
Agent 3 (Sonnet): Database schema
Agent 4 (GPT-4o-mini): Configuration and setup
```

## Model Selection Decision Tree

```
Is this a planning/analysis task?
├─ YES → Use GPT-5
└─ NO → Is this a simple/quick edit?
    ├─ YES → Use GPT-4o-mini
    └─ NO → Is this a complex feature you're confident can be one-shot?
        ├─ YES → Use Opus (if confident)
        └─ NO → Use Sonnet (safe default)
```

## Ecosystem Evolution Strategy

**Stay Current**: "New AI coding tools are coming out almost on a weekly basis"

**Recommendation**: 
- Try new tools regularly
- Evaluate against current workflow
- Integrate beneficial capabilities
- Don't change entire workflow for minor improvements

## Common Selection Mistakes

❌ **Using Opus for everything** (expensive, overkill)
❌ **Using GPT-4o-mini for complex tasks** (too weak)
❌ **Using same model for all tasks** (inefficient)
❌ **Not testing new models** (missing improvements)

✅ **Match model to task complexity**
✅ **Use fastest/cheapest model that can complete task**
✅ **Keep experimenting with new options**
✅ **Track performance and cost over time**

## Performance Tracking

**Monitor**:
- Success rate per model per task type
- Average cost per completed task
- Time to completion
- Quality of output

**Adjust**:
- Move tasks to smaller models when possible
- Upgrade to larger models when success rate drops
- Experiment with new models on non-critical tasks