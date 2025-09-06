#!/bin/bash

# 🧠 OpenAI Framework + GPT-5 Enhanced Prompt Optimizer
# Integrates proven OpenAI strategies with cutting-edge GPT-5 insights
# Based on OpenAI GPT-4.1 + GPT-5 prompting guides and Cursor's production learnings

ORIGINAL_PROMPT="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
PROJECT_ROOT=$(pwd)

# Logging setup
LOG_FILE="$HOME/.claude/logs/openai-gpt5-optimizer.log"
CACHE_DIR="$HOME/.claude/cache/openai-optimizer"
mkdir -p "$(dirname "$LOG_FILE")" "$CACHE_DIR"

log() { echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"; }

# Enhanced Analysis Engine with GPT-5 patterns
analyze_prompt_structure() {
    local prompt="$1"
    local analysis_file="$CACHE_DIR/current-analysis.json"
    
    # Enhanced analysis including GPT-5 patterns
    local prompt_length=${#prompt}
    local has_role="false"
    local has_instructions="false"
    local is_vague="false"
    local is_coding="false"
    local needs_persistence="false"
    local needs_tool_preambles="false"
    local is_complex="false"
    local is_frontend="false"
    
    [[ "$prompt" =~ (you are|act as|as a) ]] && has_role="true"
    [[ "$prompt" =~ (specifically|exactly|step by step|format) ]] && has_instructions="true"
    [[ "$prompt" =~ ^(help|fix|make|do|can you).{0,20}$ ]] && is_vague="true"
    [[ "$prompt" =~ (code|function|component|bug|debug|implement|api) ]] && is_coding="true"
    [[ "$prompt" =~ (component|ui|interface|design|frontend|react|next) ]] && is_frontend="true"
    [[ "$prompt" =~ (complex|multi|several|many|system|architecture) ]] && is_complex="true"
    [[ "$prompt" =~ (implement|build|create|develop|fix|debug) ]] && needs_persistence="true"
    [[ "$is_coding" == "true" || "$is_complex" == "true" ]] && needs_tool_preambles="true"
    
    # Create enhanced analysis
    cat > "$analysis_file" << EOF
{
    "original_length": $prompt_length,
    "has_clear_role": $has_role,
    "has_specific_instructions": $has_instructions,
    "is_vague": $is_vague,
    "is_coding_task": $is_coding,
    "is_frontend_task": $is_frontend,
    "is_complex_task": $is_complex,
    "needs_persistence": $needs_persistence,
    "needs_tool_preambles": $needs_tool_preambles,
    "timestamp": "$TIMESTAMP"
}
EOF
    
    echo "$analysis_file"
}

# GPT-5 Rule 1: Enhanced Role Definition with Agent Mindset
add_enhanced_role() {
    local prompt="$1"
    local role_section=""
    
    # GPT-5 Enhanced role definitions with agent capabilities
    if [[ "$prompt" =~ (bug|debug|error|fix|issue) ]]; then
        role_section="# Role and Objective
You are an expert debugging agent specializing in systematic problem resolution. Your mission is to autonomously identify, analyze, and implement precise solutions for code issues without requiring user intervention.

"
    elif [[ "$prompt" =~ (component|ui|interface|design|frontend|react|next) ]]; then
        role_section="# Role and Objective  
You are a senior frontend development agent with deep expertise in modern web technologies. Your goal is to autonomously create production-ready, accessible, and aesthetically excellent UI components following industry best practices.

## Recommended Technology Stack
- **Framework**: Next.js (TypeScript), React
- **Styling**: Tailwind CSS, shadcn/ui, Radix Themes
- **Icons**: Lucide, Heroicons, Material Symbols
- **Animation**: Motion (Framer Motion)
- **Fonts**: Inter, Geist, San Serif

"
    elif [[ "$prompt" =~ (api|endpoint|database|backend|server) ]]; then
        role_section="# Role and Objective
You are a backend systems architect agent. Your objective is to autonomously design and implement robust, scalable, and secure backend solutions with minimal user guidance.

"
    elif [[ "$prompt" =~ (test|testing|spec|e2e) ]]; then
        role_section="# Role and Objective
You are a test automation specialist agent. Your goal is to autonomously create comprehensive, maintainable test suites that ensure code quality and reliability.

"
    else
        role_section="# Role and Objective
You are an expert software engineering agent. Your goal is to autonomously provide precise, actionable solutions while following established coding patterns and best practices.

"
    fi
    
    echo "$role_section$prompt"
}

# GPT-5 Rule 2: Agentic Persistence Instructions (Critical for GPT-5)
add_agentic_persistence() {
    local prompt="$1"
    local persistence_section=""
    
    # Only add for tasks that need autonomous completion
    if [[ "$prompt" =~ (implement|build|create|develop|fix|debug|refactor|optimize) ]]; then
        persistence_section="
# Agentic Persistence Protocol
- **You are an autonomous agent** - keep working until the user's query is completely resolved
- **Only terminate** when you are absolutely certain the problem is solved
- **Never stop at uncertainty** - research, deduce the most reasonable approach, and continue
- **Do not ask for confirmation** - document your assumptions, act on them, and adjust if proven wrong
- **Complete all sub-tasks** - decompose the request and confirm each component is finished
- **Use tools proactively** - gather information rather than making assumptions

"
    fi
    
    echo "$prompt$persistence_section"
}

# GPT-5 Rule 3: Tool Preambles for Better User Experience
add_tool_preambles() {
    local prompt="$1"
    local preamble_section=""
    
    # Add for coding and complex tasks
    if [[ "$prompt" =~ (code|implement|build|debug|complex|multi) ]]; then
        preamble_section="
# Communication Protocol
- **Always begin** by rephrasing the user's goal in a clear, friendly manner
- **Outline your plan** immediately with structured steps before taking action
- **Narrate progress** succinctly as you execute each step
- **Provide updates** during longer operations to maintain transparency
- **Summarize completion** distinctly from your initial plan

"
    fi
    
    echo "$prompt$preamble_section"
}

# GPT-5 Rule 4: Enhanced Context Gathering Strategy
add_smart_context_gathering() {
    local prompt="$1"
    local context_section=""
    
    # Smart context strategy for coding tasks
    if [[ "$prompt" =~ (code|debug|implement|refactor) ]]; then
        context_section="
# Intelligent Context Strategy
**Goal**: Get enough context efficiently. Parallelize discovery and act as soon as possible.

**Method**:
- Start broad, then focus on specific areas
- Use targeted searches in parallel batches
- Avoid over-searching - prefer acting over endless investigation

**Early Stop Criteria**:
- You can name exact content to change
- Multiple sources converge on the same solution area
- You have enough information to make meaningful progress

**Search Depth**:
- Trace only symbols you'll modify or depend on
- Avoid transitive expansion unless necessary
- Cache discoveries to prevent duplicate work

"
    fi
    
    echo "$prompt$context_section"
}

# GPT-5 Rule 5: Self-Reflection Excellence Framework
add_self_reflection_framework() {
    local prompt="$1"
    local reflection_section=""
    
    # Add for complex or high-stakes tasks
    if [[ "$prompt" =~ (implement|build|create|architecture|system|complex) ]]; then
        reflection_section="
# Excellence Framework (Internal Use)
Before providing your solution, internally create and apply a quality rubric:

1. **Create Internal Rubric** (5-7 categories for world-class solutions)
2. **Evaluate Against Standards** - Does this meet professional production quality?
3. **Iterate Internally** - Refine until hitting top marks across all categories
4. **Deliver Excellence** - Only provide solutions that meet the highest standards

*Note: The rubric is for your internal quality control - don't show it to the user*

"
    fi
    
    echo "$prompt$reflection_section"
}

# Enhanced Instructions with GPT-5 Coding Best Practices
add_enhanced_instructions() {
    local prompt="$1"
    local instructions=""
    
    # GPT-5 Enhanced coding instructions
    if [[ "$prompt" =~ (code|function|component|implement) ]]; then
        instructions="
# Development Instructions
- **Write for clarity first** - prefer readable, maintainable solutions with clear names
- **Follow existing patterns** - study the codebase and match established conventions
- **Use TypeScript rigorously** - strict typing, proper return types, error boundaries
- **Handle edge cases** - consider error states, loading states, and boundary conditions
- **Optimize for review** - write code that's easy to quickly understand and approve

## Code Quality Standards
- Ensure all functions have proper return types and error handling
- Use meaningful variable names (avoid single letters unless in loops)
- Add comments only where business logic is complex or non-obvious
- Follow established file/folder structure patterns
- Leverage existing utilities and avoid duplication

## Frontend-Specific Guidelines (when applicable)
- **Design System**: Use shadcn/ui, Radix primitives, and Tailwind utilities
- **Accessibility**: Semantic HTML, ARIA labels, keyboard navigation
- **Performance**: Proper image optimization, lazy loading, code splitting
- **Responsive**: Mobile-first design with proper breakpoints

## Output Format Requirements
- Provide complete, working code solutions
- Use proper code blocks with language specification
- Reference file paths for context: \`filename:line_number\`
- Explain key architectural decisions briefly
- Show expected usage examples when helpful

"""
    elif [[ "$prompt" =~ (debug|bug|error|fix) ]]; then
        instructions="
# Debugging Methodology
- **Systematic approach** - follow a proven debugging process
- **Root cause focus** - identify and fix underlying issues, not symptoms
- **Minimal changes** - make targeted fixes that address the core problem
- **Verification** - test the fix thoroughly before considering complete

## Debugging Process
1. **Analyze** - Read error messages, stack traces, and understand the failure
2. **Isolate** - Identify the specific component, function, or logic causing issues
3. **Hypothesize** - Form theories about potential root causes
4. **Test** - Verify theories with minimal, reversible changes
5. **Implement** - Apply the most targeted solution
6. **Verify** - Confirm the fix resolves the issue completely

## Output Format
- Start with clear problem diagnosis and root cause analysis
- Show exact code changes with before/after context
- Explain why the fix addresses the underlying issue
- Provide verification steps to confirm resolution

"""
    else
        instructions="
# General Instructions
- **Be specific and actionable** - provide concrete, implementable solutions
- **Follow project conventions** - study existing patterns and maintain consistency
- **Include relevant context** - add background information that aids understanding
- **Provide working examples** - show practical usage when applicable

## Response Standards
- Address the request directly and completely
- Use clear, structured formatting with proper headings
- Provide working code when applicable
- Explain reasoning for key technical decisions
- Include next steps or follow-up considerations when relevant

"""
    fi
    
    echo "$prompt$instructions"
}

# Enhanced Chain of Thought for Complex Tasks
add_enhanced_chain_of_thought() {
    local prompt="$1"
    local cot_section=""
    
    if [[ "$prompt" =~ (complex|analyze|debug|plan|strategy|architecture|system) ]]; then
        cot_section="
# Systematic Reasoning Protocol
Work through this methodically:

1. **Problem Decomposition** - Break the request into specific, actionable sub-tasks
2. **Context Discovery** - Identify what information you need from the codebase
3. **Solution Architecture** - Design your approach step-by-step before coding
4. **Implementation Strategy** - Execute with careful attention to quality and consistency
5. **Verification Process** - Test and validate that your solution meets all requirements

**Important**: Show your reasoning for complex decisions, but keep it concise and focused.

"
    fi
    
    echo "$prompt$cot_section"
}

# Enhanced Project Context with GPT-5 Awareness
add_enhanced_project_context() {
    local prompt="$1"
    local context_section=""
    
    # Enhanced project detection and context
    if [[ -f "package.json" ]]; then
        local project_info=$(grep -E '"name"|"version"|"scripts"' package.json 2>/dev/null | head -3)
        if [[ -n "$project_info" ]]; then
            context_section="
# Project Context
\`\`\`json
$project_info
\`\`\`

"
        fi
        
        # Detect framework and add specific guidance
        if grep -q "next" package.json 2>/dev/null; then
            context_section="$context_section## Next.js Project Detected
- Use Next.js 14+ features (App Router, Server Components)
- Follow Next.js best practices for routing and data fetching
- Consider SSR/SSG implications for your implementation
- Use proper Image optimization and SEO practices

"
        fi
        
        if grep -q "react" package.json 2>/dev/null; then
            context_section="$context_section## React Best Practices
- Use functional components with hooks
- Implement proper error boundaries
- Follow React 18+ patterns (Suspense, Concurrent Features)
- Optimize re-renders with useMemo/useCallback when needed

"
        fi
    fi
    
    # Add existing components context for UI tasks
    if [[ "$prompt" =~ (component|ui) ]] && [[ -d "src/components" ]]; then
        local components=$(find src/components -name "*.tsx" -o -name "*.jsx" 2>/dev/null | head -5)
        if [[ -n "$components" ]]; then
            context_section="$context_section# Available Components for Reference
\`\`\`
$(echo "$components" | sed 's/^/- /')
\`\`\`

"
        fi
    fi
    
    echo "$prompt$context_section"
}

# Enhanced Output Format with GPT-5 Structured Responses
add_enhanced_output_format() {
    local prompt="$1"
    local format_section=""
    
    if [[ "$prompt" =~ (code|function|component|implement) ]]; then
        format_section="
# Expected Response Structure

Organize your response as follows:

## 1. Goal Confirmation
Brief restatement of what you're implementing and why

## 2. Implementation Plan  
- High-level approach (2-3 bullet points)
- Key technical decisions and rationale

## 3. Code Solution
\`\`\`typescript
// Well-commented, production-ready code
// Use descriptive variable names and clear structure
\`\`\`

## 4. Usage & Integration
- How to use the new code
- Integration points with existing system
- Expected behavior and results

## 5. Next Steps (if applicable)
- Follow-up tasks or considerations
- Testing recommendations
- Future enhancement opportunities

**File References**: Use format \`src/components/ComponentName.tsx:42\`

"
    elif [[ "$prompt" =~ (debug|bug|error|fix) ]]; then
        format_section="
# Debugging Response Structure

## 1. Problem Analysis
- Root cause identification
- Affected components and systems
- Impact assessment

## 2. Solution Implementation
\`\`\`typescript
// Fixed code with clear explanatory comments
// Highlight the specific changes made
\`\`\`

## 3. Verification Steps
- How to test the fix
- Expected behavior after resolution
- Additional monitoring or safeguards

## 4. Prevention
- How to avoid this issue in the future
- Code patterns or practices to adopt

"
    fi
    
    echo "$prompt$format_section"
}

# Main GPT-5 Enhanced Optimization Engine
optimize_prompt_gpt5() {
    local original="$1"
    local optimized="$original"
    
    log "🧠 Starting GPT-5 enhanced optimization..."
    
    # Skip optimization for very short or already well-structured prompts
    if [[ ${#original} -lt 20 ]] || [[ "$original" =~ ^#.*Role.*Objective ]]; then
        log "ℹ️ Prompt already optimized or too short, skipping"
        echo "$original"
        return
    fi
    
    # Apply GPT-5 enhanced framework rules in sequence
    optimized=$(add_enhanced_role "$optimized")
    optimized=$(add_agentic_persistence "$optimized")
    optimized=$(add_tool_preambles "$optimized")
    optimized=$(add_smart_context_gathering "$optimized")
    optimized=$(add_self_reflection_framework "$optimized")
    optimized=$(add_enhanced_instructions "$optimized")
    optimized=$(add_enhanced_chain_of_thought "$optimized")
    optimized=$(add_enhanced_project_context "$optimized")
    optimized=$(add_enhanced_output_format "$optimized")
    
    # Add final GPT-5 completion instructions
    optimized="$optimized
---
**Final Instructions**: Work autonomously and systematically. Use available tools to gather information rather than making assumptions. Only complete your turn when you're certain the task is fully resolved.

*🧠 Enhanced with GPT-5 Framework | OpenAI Best Practices + Agentic Intelligence*"
    
    # Save optimization results
    echo "$original" > "$CACHE_DIR/last-original.txt"
    echo "$optimized" > "$CACHE_DIR/last-optimized.txt"
    
    # Calculate improvement metrics
    local original_length=${#original}
    local optimized_length=${#optimized}
    local improvement_ratio=$(( (optimized_length * 100) / original_length ))
    
    log "✅ GPT-5 optimization complete: ${original_length} → ${optimized_length} chars (${improvement_ratio}% expansion)"
    
    echo "$optimized"
}

# Analyze prompt first
analysis_file=$(analyze_prompt_structure "$ORIGINAL_PROMPT")
log "📊 Enhanced analysis saved to: $analysis_file"

# Apply GPT-5 enhanced optimization
optimized_result=$(optimize_prompt_gpt5 "$ORIGINAL_PROMPT")

# Output the optimized prompt
echo "$optimized_result"