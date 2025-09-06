# 4 Levels of AI Coding Autonomy

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## The Autonomy Levels

### L0: Fully Manual
- Humans do all the work
- No AI assistance
- Traditional programming approach

### L1: Human Assisted  
- AI provides completions and suggestions
- Copy-paste from ChatGPT
- Tools: GitHub Copilot, basic ChatGPT usage
- Era: 3-4 years ago

### L2: Human Monitored (VIP Coding)
- AI handles most tasks, humans watch for issues
- Current primary stage for most developers
- Tools: Replit, Cursor, Lovable, v0, Windsurf
- Era: Started ~1 year ago

### L3: Human Out of the Loop (Agentic Coding)
- AI handles everything start to finish
- Humans only review PRs
- Tools: Claude Code, Cursor background agents
- Era: Current cutting edge

## The Critical Mistake

**Problem**: People skip L0, L1, L2 and jump straight to L3
- Create 20+ Claude Code sub-agents immediately
- Ask AI to create agents for them
- Result: "Manage to ship absolutely nothing"

**Context Loss Issue**:
- Each new agent loses context from previous tasks
- Key decisions, reasoning, next steps are lost
- Multiple agents working on same codebase with no memory
- "Like trying to go from city A to city B on different trains, but every single train takes you in a completely random direction"

## Tool Selection by Level

### L2 (VIP Coding) Tools:
- Replit
- Cursor
- Lovable  
- v0
- Windsurf

### L3 (Agentic Coding) Tools:
- Claude Code
- Cursor background agents

## Why Backend is Harder

**Frontend**: Established frameworks (React, Next.js) provide structure
**Backend**: No unified architecture patterns
- Event-driven for latency/UX
- Microservices for scale
- Monolith for internal projects
- Many more options and patterns

**Result**: "This is why Lovable started from front ends and not from backends"

## Progression Strategy

1. **Master L2 first** - Learn to work with AI effectively
2. **Build foundation** - Understand architectures and patterns  
3. **Graduate to L3** - Only after mastering supervision and context management
4. **Use proper workflow** - Don't skip the 5-step process

## Key Insight

**The Problem**: "Anytime you start a new agent it inevitably loses some context from the previous tasks. All the key moments and decisions, reasoning and next steps are unavoidably going to be lost."

**The Solution**: Proper architecture planning + documentation (ADRs) + the 5-step workflow prevents this context loss.