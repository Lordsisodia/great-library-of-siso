# Context Management Strategies for AI Coding

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## The Core Principle

**Golden Rule**: "Always think about what your AI can and cannot see"

**Context Problems**:
- **Not enough context** = AI hallucinates
- **Too much context** = AI gets overwhelmed
- **Wrong context** = AI solves wrong problem

## The Self-Check Method

**Before every prompt, ask yourself**:
> "Would I be able to complete this task with the context that I've given to AI?"

**If NO** → Add more relevant context
**If YES** → You're ready to proceed

## Context Loss Prevention

### 1. Write Tests First (While Context is Complete)
- AI loses context over long conversations
- Coding assistants remove middle/beginning messages
- Tests written early preserve the original intent
- **Result**: "AI literally can't fool itself because it's going to run the test"

### 2. Use ADR Documentation
- **ADR** = Architecture Decision Records
- Document WHY decisions were made, not just what
- Preserve context for future agents
- **Quote**: "Right context at the right time"

### 3. Break Complex Features Early
- Long conversations = inevitable context loss
- Split complex features into smaller tasks
- Each task should be completable with current context

## Context Window Optimization

### What AI Needs to See:
- **Relevant files** (not entire codebase)
- **Type definitions** for current feature
- **Related functions/components** being modified
- **Test examples** for similar functionality
- **Error messages** from previous attempts

### What AI Doesn't Need:
- **Entire project history**
- **Unrelated files**
- **Old conversation threads**
- **General documentation** (unless specifically relevant)

## Practical Context Strategies

### 1. File Selection Strategy
```
# Good Context
- Current component being modified
- Type definitions it uses
- Related test files
- Specific error messages

# Bad Context  
- Entire src/ directory
- All test files
- Full git history
- Multiple unrelated components
```

### 2. Progressive Context Building
```
Step 1: Show AI the main file to modify
Step 2: Add relevant types/interfaces  
Step 3: Include related functions
Step 4: Add test examples
Step 5: Include error messages if any
```

### 3. Context Reset Indicators
**When to reset context**:
- Conversation > 20 messages
- AI starts hallucinating obvious things
- Multiple failed attempts at same task
- Agent suggests non-existent files/functions

## Advanced Context Techniques

### 1. Context Anchoring with Types
- Types act as "anchors" preventing hallucination
- Define complete type system before implementation
- AI uses types as guardrails for logic

### 2. Context Preservation via Documentation
```
# In ADR document:
## Context for Future Agents
- Why we chose Firebase over custom backend
- Why event-driven architecture was selected  
- Key issues encountered with 11Labs API
- Solutions that worked/didn't work
```

### 3. Multi-Agent Context Sharing
- Use git changes to track what each agent is working on
- Reference completed tasks in new agent prompts
- Share types and architecture decisions across agents

## Context Management in 5-Step Workflow

**Step 1 (Architecture)**: Establish context foundation
**Step 2 (Types)**: Create context anchors
**Step 3 (Tests)**: Preserve context for future
**Step 4 (Implementation)**: Use established context
**Step 5 (Documentation)**: Save context for next cycle

## Red Flags: Context Issues

🚩 **AI suggests non-existent libraries**
🚩 **AI references functions that don't exist**
🚩 **AI ignores constraints you mentioned earlier**
🚩 **AI asks for information you already provided**
🚩 **AI solutions don't match established architecture**

## Recovery Strategies

**When Context is Lost**:
1. **Reset conversation** with fresh context
2. **Reference ADR documents** from previous work
3. **Include relevant test files** as context
4. **Re-establish type definitions**
5. **Reference working examples** from codebase

**Prevention Better Than Cure**: Use the 5-step workflow to prevent context loss rather than trying to recover from it.