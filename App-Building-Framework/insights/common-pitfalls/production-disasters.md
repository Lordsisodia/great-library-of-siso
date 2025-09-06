# Production Disasters and How to Avoid Them

**Source**: [AI Coding Workflow YouTube Video](../videos/youtube/ai-coding-workflow-production-template.md)

## Real Production Disasters

### 1. The Replit Database Deletion (Saster)
**What Happened**: 
- Simon Sherwood (CEO of Saster) was vibe coding with Replit
- Day 8: Replit deleted their entire production database
- Saster is a large company - would have been expensive for Replit
- Initially told recovery was impossible, but later recovered

**Root Cause**: AI agent connected directly to production database

### 2. The Accept-All Security Trap
**Statistics**:
- 20% of AI-generated code recommends non-existing libraries  
- Only 55% of AI-generated code passes basic security tests

**Root Cause**: "AI is still lazy. It doesn't really understand that it's coding in production. It thinks this is a game or another benchmark."

### 3. Technical Debt Explosion
**Stanford Study Results**:
- Developers produce 30-40% more code with AI
- Significant portion needs reworking
- Net productivity gain: only 15-20%

**Root Cause**: Temptation to push code as soon as feature appears complete

## Prevention Strategies

### 1. Never Connect AI to Production Database
**Solution**: Create staging environment
- Exact same project setup on AWS/Google Cloud
- Same configuration as production
- Uses empty database instead of real user data
- **Simple Rule**: AI agents only touch staging, never production

### 2. Use Modern Tech Stack with Built-in Security
**Recommended Stacks**:
- **Supabase** - Built-in security rules
- **Firebase** - Simplified authentication patterns

**Firebase Example**: `onCall` function handles all authentication automatically
- "Literally no way that you can mess up authentication because it's all handled by Firebase"

### 3. Implement "No Broken Windows" Rule
**The Rule**: Whenever something can be improved, it needs to be improved right away

**Why This Works**:
- Prevents accumulation of technical debt
- Stops the temptation to push unfinished code
- Maintains code quality standards consistently

**Implementation**: Add this rule to all coding assistant prompts

## Over-Engineering Pitfall

### The Framework Trap
**Common Mistake**: Using complicated frameworks on top of other frameworks
- Examples mentioned: CloudFlow, Agent OS, the BMAT method
- These create prompts on top of existing AI coding assistants

**The Problem**:
- Coding assistants already have tons of internal prompts
- Adding more prompts makes steering much harder
- "Almost impossible to change the direction of the project"
- AI designs everything itself, user loses control

**Solution**: Use tools directly, not meta-frameworks on top of tools

## Context Loss Disasters

### The Multiple Agent Problem
**What Happens**:
- People create 20+ Claude Code sub-agents
- Or ask AI to create agents for them
- Each agent loses context from previous agents
- Result: "Manage to ship absolutely nothing"

**The Train Metaphor**: "Like trying to go from city A to city B on different trains, but every single train takes you in a completely random direction"

### Why This Happens
- New agents have no memory of previous work
- Key decisions and reasoning are lost
- Agents work on same codebase with different assumptions
- No coordination between agent sessions

## Security Disaster Prevention

### Authentication Best Practices
- Use managed authentication services
- Don't build custom auth with AI
- Leverage platform security features
- Test security rules before deployment

### Database Security
- Never grant AI direct production access
- Use staging environments for all AI development
- Implement proper access controls
- Regular security audits of AI-generated code

### Code Review Requirements
- Review all AI-generated security-related code
- Test authentication flows manually
- Validate API security before deployment
- Use security-first tech stacks

## Recovery Strategies

### When Disaster Strikes
1. **Immediate**: Disconnect AI from production systems
2. **Assessment**: Identify scope of damage
3. **Recovery**: Use backups, staging environments to rebuild
4. **Prevention**: Implement proper safeguards
5. **Documentation**: Record in ADR what went wrong and why

### Building Resilient Systems
- Always have staging environment
- Regular backups of production data
- Clear rollback procedures
- Monitoring and alerting systems
- Incident response procedures

## Red Flags to Watch For

🚩 **AI agent has production database access**
🚩 **No staging environment setup**
🚩 **Accepting all AI suggestions without review**
🚩 **Using multiple meta-frameworks**
🚩 **No security testing of AI-generated code**
🚩 **Pushing code immediately after feature completion**
🚩 **Creating 20+ agents without coordination**

## Daily Prevention Habits

✅ **Always use staging for AI development**
✅ **Review security-related code manually**  
✅ **Implement "no broken windows" rule**
✅ **Use simple, proven tech stacks**
✅ **Document architectural decisions in ADRs**
✅ **Test authentication flows manually**
✅ **Regular backups and recovery testing**