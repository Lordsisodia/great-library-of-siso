# Multi-Agent Coordination for AI Coding

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## The Multi-Agent Strategy

**Core Concept**: "You can create parallel agents for any tasks that don't have any dependencies"

**Productivity Impact**: "Can you just imagine how much of an increase in productivity you get by launching four agents in parallel?"

**Future Vision**: "This is, I think, how coding is going to evolve in the future. You're simply going to be supervising multiple agents at the same time."

## Agent Assignment Strategy

### Dependency-Based Assignment
**Rule**: Only run agents in parallel when their tasks have no dependencies

**Example from Video**:
```
Agent 1: 11Labs API Wrapper (no dependencies)
Agent 2: FFmpeg Video Service (no dependencies)  
Agent 3: Firestore Security Rules (no dependencies)
Agent 4: Frontend Auth Protection (no dependencies)

Later (after dependencies complete):
Agent 5: Background Processing (depends on Agent 1,2)
Agent 6: Upload Form UI (depends on Agent 3,4)
```

### Tool-Specific Agent Roles
**Backend Agents**: Cursor with Sonnet
- API integrations
- Service implementations  
- Database schema
- Background processing

**Frontend Agents**: Claude Code
- UI components
- Authentication flows
- Real-time updates
- Form handling

## Coordination Techniques

### 1. Git Changes Monitoring
**Primary tracking method**: "My favorite way of keeping track of changes when working with multiple agents is to simply use the git changes"

**Benefits**:
- See all files being modified across agents
- Detect potential conflicts early
- Monitor progress across all parallel work
- **Real impact**: "40 different files modified in less than 10 minutes"

### 2. Task Handoff Process
```markdown
## Agent Handoff Checklist
- [ ] Current agent completes assigned task
- [ ] Integration tests pass with real data
- [ ] Relevant files committed to git
- [ ] Next agent receives dependency context
- [ ] Architecture decisions documented in ADR
```

### 3. Context Sharing Between Agents
**Method**: Reference completed tasks in new agent prompts
```
IMPLEMENT triggered function for background processing
DEPENDENCIES COMPLETED:
- 11Labs API wrapper (Agent 1) 
- FFmpeg service (Agent 2)
- Firestore rules (Agent 3)
REFERENCE: Use types and patterns from completed tasks
```

## Quality Control Across Agents

### Focus on Test Review
**Strategy**: "Review only the tests"
**Reason**: "If the test is correct, then there is very little chance that AI can actually screw up the functionality"

**What to check per agent**:
- Are tests using real data (not mocks)?
- Do integration tests call actual APIs?
- Are tests properly isolated from other agents' work?

### Real Data Testing Requirement
**Critical Rule**: "Tell it to test it with some real data"

**Examples**:
- **11Labs Agent**: Must test with real API keys and voice generation
- **FFmpeg Agent**: Must test with actual video file processing  
- **Frontend Agent**: Must test with real Firebase connection

## Agent Communication Patterns

### Sequential Dependencies
```
Agent A completes → Updates shared state → Agent B starts with context
```

**Implementation**:
1. Agent A documents completion in ADR
2. Agent A commits all changes to git
3. Agent B references Agent A's work in initial prompt
4. Agent B validates integration with Agent A's output

### Parallel Independence  
```
Agent A ←→ Shared Architecture ←→ Agent B
```

**Requirements**:
- Both agents reference same architecture (ADR)
- Both agents use same type definitions  
- Both agents follow same coding patterns
- Minimal overlap in file modifications

## Real-World Coordination Example

### Video Processing System (from video):

**Wave 1 (Parallel - No Dependencies)**:
- **Agent 1**: 11Labs API integration + tests
- **Agent 2**: FFmpeg video processing + tests  
- **Agent 3**: Firestore/Storage security rules
- **Agent 4**: Frontend authentication + routing

**Wave 2 (Sequential - Has Dependencies)**:
- **Agent 5**: Background processing function (needs Agent 1,2)
- **Agent 6**: Upload UI with status updates (needs Agent 3,4)

**Wave 3 (Integration)**:
- **Agent 7**: End-to-end integration testing
- **Agent 8**: Deployment and production config

### Results:
- **Speed**: "Project almost fully completed in under an hour"
- **Quality**: Integration tests passing with real APIs
- **Scale**: "40+ files modified across multiple agents"

## Supervision Strategies

### Human Role Evolution
**Quote**: "Programming skills can still be useful. It's not in actually writing the code. It's in being able to review the code and make sure that the AI simply hasn't hallucinated."

### Active Supervision Tasks:
1. **Monitor Direction**: Check agents aren't going off-track
2. **Review Integration**: Ensure agents work together properly
3. **Validate Tests**: Confirm tests use real data and scenarios
4. **Manage Dependencies**: Coordinate agent sequencing
5. **Update Context**: Share learnings between agents

### Red Flags to Watch For:
🚩 **Agent creating conflicting architecture**
🚩 **Agent using mock data when real data is available**
🚩 **Agent ignoring established patterns from other agents**
🚩 **Agent modifying files being worked on by another agent**
🚩 **Agent hallucinating about non-existent functions/libraries**

## Tool-Independent Workflow

**Core Principle**: "This workflow is actually fully AI tool independent"

**Adaptation for Different Tools**:
- **Cursor users**: Reference agents.md file with AI rules
- **Claude Code users**: Use direct task assignment
- **Other tools**: Adapt prompting patterns but keep coordination strategy

### Universal Coordination Files:
```
project/
├── tasks/
│   ├── task-01-api-integration.md
│   ├── task-02-video-processing.md
│   └── task-03-security-rules.md
├── docs/
│   ├── ADR.md (Architecture Decision Records)
│   └── agents.md (AI agent rules and patterns)
└── types/
    └── shared-interfaces.ts
```

## Advanced Coordination Patterns

### Agent Specialization
**Backend Specialists**: Handle API integrations, data processing
**Frontend Specialists**: Handle UI, authentication, real-time updates  
**DevOps Specialists**: Handle deployment, security, monitoring

### Agent Memory Sharing
```markdown
## Shared Agent Memory (ADR)
### Completed by Agent 1:
- 11Labs integration working with voice ID system
- Error handling for API rate limits implemented
- Integration tests passing with real API

### Context for Future Agents:
- Use voice ID validation pattern from Agent 1
- API rate limiting strategy documented  
- Real API testing setup available
```

### Conflict Resolution
**When agents conflict**:
1. **Pause both agents**
2. **Review git changes** to understand conflict
3. **Update shared context** (ADR, types) with resolution
4. **Restart one agent** with conflict resolution context
5. **Continue other agent** once resolution implemented

This multi-agent approach transforms development from linear coding to parallel supervision, dramatically increasing development speed while maintaining code quality through proper coordination and testing strategies.