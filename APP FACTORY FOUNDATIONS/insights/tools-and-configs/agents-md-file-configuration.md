# Agents.md File Configuration for AI Tools

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## Why agents.md is Critical

**Core Purpose**: Provides consistent AI rules across different coding tools when not using Cursor

**Quote**: "This workflow is actually fully AI tool independent. The only thing is that if you're using any other tool but cursor, you simply need to reference the agents.md file which references all of the other rules for AI agents."

## What agents.md Contains

### AI Agent Rules
- Coding standards and patterns to follow
- Architecture guidelines specific to your project
- Security rules and best practices
- Testing requirements and standards
- File organization patterns

### Project-Specific Context
- Technology stack decisions
- API integration patterns
- Database schema guidelines
- Authentication and authorization rules

## Template Integration Strategy

### Repository Structure
```
project/
├── agents.md                    # AI rules and guidelines
├── tasks/                       # Task templates
│   ├── task-template.md
│   └── individual-tasks.md
├── docs/
│   ├── ADR.md                  # Architecture Decision Records
│   └── PRD.md                  # Product Requirements Document
└── types/
    └── shared-interfaces.ts    # Type definitions
```

### Template Benefits
**Quote**: "What's really cool about having a PRD and all of the task templates directly inside your repo is that you have full control over them"

**vs External Tools**: 
- **Problem**: "When you use repos like Taskmaster, you don't really have the templates that they're using for their tasks, and so you have way less control over how your agents are scoping the tasks themselves"
- **Solution**: "If all of your templates for all the AI rules are in the same repo, then you can simply adjust them for whichever workflow you prefer"

## agents.md Content Structure

### 1. Workflow Rules
```markdown
## 5-Step AI Coding Workflow
1. Architecture Planning (types, structure, ADR)
2. Type Definitions (request, response, database types)
3. Test Generation (integration tests with real data)
4. Feature Implementation (parallel when possible)
5. Documentation Updates (ADR with key decisions)
```

### 2. Code Quality Standards
```markdown
## No Broken Windows Rule
- Fix issues immediately when discovered
- Never push incomplete or broken code
- Improve anything that can be improved right away
```

### 3. Testing Requirements
```markdown
## Testing Standards
- Integration tests over unit tests during development
- Use real APIs and real data (not mocks)
- Test with actual files in /test-data/ folder
- Always run tests and verify real outputs exist
```

### 4. Information Dense Keywords
```markdown
## Prompting Guidelines
Use these keywords for clarity:
- CREATE: Build new functionality
- UPDATE: Modify existing code
- DELETE: Remove code/features  
- ADD: Append new elements
- REMOVE: Take away specific elements
- IMPLEMENT: Execute planned tasks
```

### 5. Model Selection Guidelines
```markdown
## Model Usage Rules
- GPT-4o-mini: Quick edits and simple tasks
- GPT-5: Analysis, planning, and scoping
- Sonnet: General coding workflows (preferred)
- Opus: One-shot complex features only
```

## Tool-Independent Implementation

### For Cursor Users
- Built-in workflow integration
- Direct task assignment
- Automatic rule application

### For Other Tools
- Reference agents.md in every prompt
- Include task templates in prompts
- Manually enforce workflow steps

### Universal Prompt Pattern
```
IMPLEMENT [task name]

REFERENCE: 
- agents.md (project AI rules)
- task-[number].md (specific task)
- ADR.md (architectural context)

FOLLOW: 5-step workflow exactly
```

## Real-World Usage Example

### Before Task Execution
```markdown
IMPLEMENT authentication and route protection

DEPENDENCIES: None (parallel with other tasks)

REFERENCE FILES:
- agents.md: AI coding rules
- ADR.md: Architecture decisions
- types/auth.ts: Authentication types

REQUIREMENTS:
- Follow Firebase authentication patterns
- Use real Firebase connection (not mocks)
- Add integration tests with real user flow
- Document any issues in ADR.md
```

### During Multi-Agent Coordination
```markdown
## Agent Handoff Context

PREVIOUS AGENT COMPLETED:
- 11Labs API integration (Agent 1)
- FFmpeg video processing (Agent 2) 
- Firebase security rules (Agent 3)

NEXT AGENT CONTEXT:
- Use established patterns from agents.md
- Reference completed types from previous agents
- Follow same architecture decisions in ADR.md
- Test integration with real APIs from previous work
```

## Quality Control Integration

### Consistent Standards
- All agents follow same coding patterns
- Architecture stays consistent across parallel work
- Security rules applied uniformly
- Testing standards maintained

### Context Preservation
- agents.md captures institutional knowledge
- ADR.md documents specific decisions
- Type definitions provide concrete guardrails
- Task templates ensure complete scope coverage

## Template Customization Strategy

### Project-Specific Rules
```markdown
## Firebase-Specific Guidelines
- Use onCall functions for authentication
- Implement event-driven architecture patterns
- Follow Firestore security rules exactly
- Deploy with single command: firebase deploy
```

### Technology Stack Rules
```markdown
## Stack-Specific Patterns
- Frontend: React + TypeScript + Tailwind
- Backend: Firebase Functions + Node.js
- Database: Firestore with security rules
- Storage: Firebase Storage with access controls
```

This agents.md approach ensures consistent, high-quality AI coding across any tool while maintaining full control over your development workflow and architectural decisions.