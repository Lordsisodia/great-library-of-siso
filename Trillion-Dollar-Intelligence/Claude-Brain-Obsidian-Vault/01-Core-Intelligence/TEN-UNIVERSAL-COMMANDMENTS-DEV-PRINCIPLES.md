# The Ten Universal Commandments - Development Principles

**Source**: Development Philosophy & Architecture Guidelines
**Context**: Fundamental principles for AI-assisted development and code quality
**Purpose**: Core reference for consistent, maintainable, and principled development

---

## 🏗️ Architecture Decisions (Consider Implications)

### Use Git Tools For:
- **Before modifying files** (understand history)
- **When tests fail** (check recent changes) 
- **Finding related code** (git grep)
- **Understanding features** (follow evolution)
- **Checking workflows** (CI/CD issues)

---

## ⚖️ The Ten Universal Commandments

### **1. Thou shalt ALWAYS use MCP tools before coding**
*Never start development without leveraging available Model Context Protocol capabilities*

### **2. Thou shalt NEVER assume; always question**
*Question requirements, assumptions, and existing patterns before implementation*

### **3. Thou shalt write code that's clear and obvious**
*Self-documenting code > clever code. Clarity is king.*

### **4. Thou shalt be BRUTALLY HONEST in assessments**
*Honest evaluation of complexity, risks, and capabilities. No sugar-coating.*

### **5. Thou shalt PRESERVE CONTEXT, not delete it**
*Context is precious. Maintain it, enhance it, never destroy valuable information.*

### **6. Thou shalt make atomic, descriptive commits**
*Each commit should do one thing well with clear explanation of what and why.*

### **7. Thou shalt document the WHY, not just the WHAT**
*Code shows what you did, comments should explain why you did it.*

### **8. Thou shalt test before declaring done**
*No feature is complete without verification. Test thoroughly before claiming success.*

### **9. Thou shalt handle errors explicitly**
*Error handling is not optional. Plan for failure modes and handle them gracefully.*

### **10. Thou shalt treat user data as sacred**
*User data privacy and security are non-negotiable. Protect it with utmost care.*

---

## 📋 Final Reminders

### **Priority Hierarchy (Truth Order)**
1. **Codebase** - The source of truth
2. **Documentation** - Current and maintained docs
3. **Training data** - Historical patterns and knowledge

### **Development Workflow**
- **Research current docs** - Don't trust outdated knowledge
- **Ask questions early and often** - Clarify before implementing
- **Use slash commands** - Leverage consistent workflows
- **Derive documentation on-demand** - Create docs as needed
- **Extended thinking** - Use for complex problems
- **Visual inputs** - Screenshots for UI/UX debugging
- **Test locally** - Always verify before pushing
- **Think simple** - Clear, obvious, no bullshit approach

---

## 💡 Core Philosophy

### **The Golden Rule**
> **"Write code as if the person maintaining it is a violent psychopath who knows where you live. Make it that clear."**

### **Principles in Action**
- **Clarity over Cleverness** - Obvious solutions win
- **Context Preservation** - Information is valuable, preserve it
- **Honest Assessment** - Brutal truth about complexity and capabilities
- **User-Centric Security** - Treat user data as sacred
- **Test-Driven Completion** - Not done until tested
- **Question Everything** - Never assume, always verify

---

## 🔗 Integration with Claude Brain System

### **MCP Tools Alignment**
These commandments align perfectly with our advanced MCP setup:
- **Serena MCP** - Code analysis and understanding (Commandment #1, #2)
- **Zen MCP** - Multi-model consensus for complex decisions (Commandment #4)
- **Sequential Thinking** - Extended reasoning for complex problems (Commandment #8)

### **Community Workflows Integration**
- **Phase-based development** supports atomic commits (Commandment #6)
- **20% context rule** preserves context (Commandment #5)
- **Handoff templates** document the WHY (Commandment #7)

### **Quality Standards**
- **95%+ success rate** requires testing before done (Commandment #8)
- **Pattern consistency** demands clear code (Commandment #3)
- **Error handling** must be explicit (Commandment #9)

---

## 📊 Application Checklist

### **Before Starting Any Development**
- [ ] **MCP Tools Ready?** (Commandment #1)
- [ ] **Requirements Questioned?** (Commandment #2)
- [ ] **Context Preserved?** (Commandment #5)

### **During Development**
- [ ] **Code is Clear?** (Commandment #3)
- [ ] **Honest About Complexity?** (Commandment #4)
- [ ] **WHY Documented?** (Commandment #7)
- [ ] **Errors Handled?** (Commandment #9)
- [ ] **User Data Protected?** (Commandment #10)

### **Before Completion**
- [ ] **Thoroughly Tested?** (Commandment #8)
- [ ] **Atomic Commits Made?** (Commandment #6)
- [ ] **Context Maintained?** (Commandment #5)

---

## 🎯 Quick Reference Commands

### **Git Analysis (Architecture Decisions)**
```bash
# Understanding history before modification
git log --oneline -10
git blame filename.ext
git show HEAD~1

# Finding related code  
git grep "pattern"
git log --grep="feature"

# Checking recent changes when tests fail
git diff HEAD~1
git show --stat
```

### **MCP Integration Commands**
```bash
# Always start with MCP analysis
"Read Serena's initial instructions and analyze this project structure"
"Use zen's planner with Gemini Pro for complex architectural decisions"
"Apply sequential thinking for multi-step reasoning"
```

---

## 📝 Memory Triggers

### **Daily Reminders**
- **Start with MCP** - Never code blind
- **Question First** - Assumptions kill projects
- **Clear Code** - Future you will thank you
- **Preserve Context** - Information is precious
- **Test Everything** - Untested code is broken code

### **Red Flags to Avoid**
- Skipping MCP tool analysis
- Making assumptions without verification
- Writing clever but unclear code
- Deleting valuable context
- Committing without testing
- Ignoring error scenarios
- Compromising user data protection

---

**Tags**: #development-principles #ten-commandments #architecture-decisions #code-quality #mcp-integration
**Related**: [[CLAUDE-INTELLIGENCE-UPGRADE]] [[SISO-ULTRA-THINK-SYSTEM]] [[Reddit-Post-CLAUDE-md-Structure-Analysis]]
**Status**: ✅ Core Reference | 🎯 Daily Application Ready