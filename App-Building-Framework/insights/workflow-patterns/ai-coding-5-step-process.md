# AI Coding 5-Step Production Workflow

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## The Core 5-Step Process

### 1. Architecture Planning (BEFORE any code)
- **PRD (Product Requirements Document)** - Well-defined requirements with user stories
- **Project Structure** - Clean architecture from the start
- **ADR (Architecture Decision Records)** - Document WHY decisions were made, not just what
- **Workflow Documentation** - Clear instructions for AI agents

**Key Quote**: "When you get architecture right from the start, something magical happens. You don't need hundreds of programmers or 24/7 coding to keep it working. It just does it."

### 2. Create Types First
- Define ALL types before writing any functionality
- Include: request types, response types, component types, database types
- Types act as "anchors" that prevent AI hallucination
- Linters catch type errors immediately, providing feedback loop

**Why This Works**: "If the agent has defined all the types for the feature correctly, there's actually very little room for this agent to hallucinate and make a mistake."

### 3. Generate Tests (Most Important Step)
- Write tests FIRST while context is still complete
- AI loses context over long conversations - tests prevent this
- Integration tests > Unit tests during development
- Tests with real data, not mock data
- AI "literally can't fool itself" when tests exist

**Critical Insight**: "When you define these three things first - the types, the tests, and the architecture - you essentially built a railroad for AI to complete this feature reliably."

### 4. Build the Feature
- Minimal prompts needed if steps 1-3 done correctly
- Can run multiple agents in parallel on different parts
- AI "just runs by itself until the feature is fully completed"
- Focus on reviewing tests, not all code changes

### 5. Document Changes
- Update ADR with key architectural decisions
- Document context for next agents
- "Right context at the right time" principle
- Prevents context loss between agent sessions

## Multi-Agent Coordination

**Parallel Agent Strategy**:
- Launch 4+ agents simultaneously on non-dependent tasks
- Track changes via git changes panel
- Review only the tests, not all code
- Supervise direction, don't write code

**Productivity Gains**: "Before for a real developer, this would have taken like a week to do by himself. While now I'm literally coding this and it's not even been like an hour and the project is almost fully completed."

## Context Management Rules

**The Railroad Principle**: Architecture + Types + Tests = AI cannot go sideways
- "There is simply no way that AI can fail if these three things are correct"
- Well-architected projects need only minimal prompts
- AI completes features autonomously within defined boundaries

**Memory Management**:
- ADRs preserve decision context across agent sessions
- Document issues and solutions immediately
- Future agents learn from previous agent mistakes

## Implementation Notes

- Start with Firebase for simplified architecture
- Use event-driven patterns when possible
- Always test with real data/APIs
- Deploy frequently to catch integration issues early