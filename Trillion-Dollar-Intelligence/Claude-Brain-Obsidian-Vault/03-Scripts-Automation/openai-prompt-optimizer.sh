#!/bin/bash

# 🧠 OpenAI Framework-Based Prompt Optimizer
# Applies OpenAI's proven prompting strategies to rewrite and optimize prompts for Claude Code
# Based on GPT-4.1 prompting guide and best practices research

ORIGINAL_PROMPT="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
PROJECT_ROOT=$(pwd)

# Logging setup
LOG_FILE="$HOME/.claude/logs/openai-optimizer.log"
CACHE_DIR="$HOME/.claude/cache/openai-optimizer"
mkdir -p "$(dirname "$LOG_FILE")" "$CACHE_DIR"

log() { echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"; }

# OpenAI Framework Analysis Engine
analyze_prompt_structure() {
    local prompt="$1"
    local analysis_file="$CACHE_DIR/current-analysis.json"
    
    # Simple analysis without complex bash substitution
    local prompt_length=${#prompt}
    local has_role="false"
    local has_instructions="false"
    local is_vague="false"
    local is_coding="false"
    
    [[ "$prompt" =~ (you are|act as|as a) ]] && has_role="true"
    [[ "$prompt" =~ (specifically|exactly|step by step|format) ]] && has_instructions="true"
    [[ "$prompt" =~ ^(help|fix|make|do|can you).{0,20}$ ]] && is_vague="true"
    [[ "$prompt" =~ (code|function|component|bug|debug|implement|api) ]] && is_coding="true"
    
    # Create analysis file
    cat > "$analysis_file" << EOF
{
    "original_length": $prompt_length,
    "has_clear_role": $has_role,
    "has_specific_instructions": $has_instructions,
    "is_vague": $is_vague,
    "is_coding_task": $is_coding,
    "timestamp": "$TIMESTAMP"
}
EOF
    
    echo "$analysis_file"
}

# OpenAI Rule 1: Clear Role and Objective
add_clear_role() {
    local prompt="$1"
    local role_section=""
    
    # Detect task type and add appropriate role
    if [[ "$prompt" =~ (bug|debug|error|fix|issue) ]]; then
        role_section="# Role and Objective
You are an expert debugging assistant specializing in software engineering. Your primary goal is to identify, analyze, and provide precise solutions for code issues.

"
    elif [[ "$prompt" =~ (component|ui|interface|design) ]]; then
        role_section="# Role and Objective  
You are a senior frontend developer and UX engineer. Your goal is to create clean, accessible, and well-structured UI components following modern best practices.

"
    elif [[ "$prompt" =~ (api|endpoint|database|backend) ]]; then
        role_section="# Role and Objective
You are a backend systems architect. Your objective is to design and implement robust, scalable, and secure backend solutions.

"
    elif [[ "$prompt" =~ (test|testing|spec) ]]; then
        role_section="# Role and Objective
You are a test automation specialist. Your goal is to create comprehensive, maintainable test suites that ensure code quality and reliability.

"
    else
        role_section="# Role and Objective
You are an expert software engineering assistant. Your goal is to provide precise, actionable solutions while following established coding patterns and best practices.

"
    fi
    
    echo "$role_section$prompt"
}

# OpenAI Rule 2: Specific Instructions with Delimiters
add_specific_instructions() {
    local prompt="$1"
    local instructions=""
    
    # Add coding-specific instructions
    if [[ "$prompt" =~ (code|function|component|implement) ]]; then
        instructions="
# Instructions
- Follow existing code patterns and conventions in the codebase
- Use TypeScript strict typing where applicable  
- Include error handling and edge case considerations
- Write clean, readable code with meaningful variable names
- Add inline comments for complex logic only

## Code Quality Requirements
- Ensure all functions have proper return types
- Handle async operations with proper error boundaries
- Follow the established file/folder structure
- Use existing utility functions and avoid duplication

## Output Format
- Provide complete, working code solutions
- Use proper code blocks with language specification
- Include file paths for context: \`filename:line_number\`
- Explain key decisions in brief comments

"""
    elif [[ "$prompt" =~ (debug|bug|error|fix) ]]; then
        instructions="
# Instructions  
- Analyze the problem systematically using a debugging methodology
- Identify root cause rather than treating symptoms
- Provide step-by-step troubleshooting approach
- Test hypotheses before implementing fixes

## Debugging Process
1. **Understand**: Read error messages and stack traces carefully
2. **Isolate**: Identify the specific component or function causing issues  
3. **Hypothesize**: Form theories about potential causes
4. **Test**: Verify theories with minimal code changes
5. **Fix**: Implement the most targeted solution
6. **Verify**: Confirm the fix resolves the issue completely

## Output Format
- Start with problem analysis
- Show exact code changes needed
- Explain why the fix addresses the root cause

"""
    else
        instructions="
# Instructions
- Be specific and actionable in your response
- Provide concrete examples where helpful
- Follow established project patterns and conventions
- Include relevant context and background information

## Response Requirements
- Address the request directly and completely
- Use clear, structured formatting
- Provide working code when applicable
- Explain reasoning for key decisions

"""
    fi
    
    echo "$prompt$instructions"
}

# OpenAI Rule 3: Chain of Thought for Complex Tasks
add_chain_of_thought() {
    local prompt="$1"
    local cot_section=""
    
    if [[ "$prompt" =~ (complex|analyze|debug|plan|strategy|architecture) ]]; then
        cot_section="
# Reasoning Strategy
Think through this systematically:

1. **Problem Analysis**: Break down the request into specific sub-tasks
2. **Context Gathering**: Identify what information is needed from the codebase
3. **Solution Planning**: Outline the approach step-by-step  
4. **Implementation**: Execute the plan with careful attention to details
5. **Verification**: Check that the solution addresses all requirements

Please work through each step explicitly before providing your final solution.

"
    fi
    
    echo "$prompt$cot_section"
}

# OpenAI Rule 4: Examples and Context
add_examples_and_context() {
    local prompt="$1"
    local context_section=""
    
    # Add project-specific context
    if [[ -f "package.json" ]]; then
        local project_info=$(grep -E '"name"|"version"' package.json 2>/dev/null | head -2)
        if [[ -n "$project_info" ]]; then
            context_section="
# Project Context
Current project details:
\`\`\`json
$project_info
\`\`\`

"
        fi
    fi
    
    # Add file structure context for component/UI tasks
    if [[ "$prompt" =~ (component|ui) ]] && [[ -d "src/components" ]]; then
        local components=$(find src/components -name "*.tsx" -o -name "*.jsx" 2>/dev/null | head -5)
        if [[ -n "$components" ]]; then
            context_section="$context_section# Existing Components for Reference
\`\`\`
$(echo "$components" | sed 's/^/- /')
\`\`\`

"
        fi
    fi
    
    echo "$prompt$context_section"
}

# OpenAI Rule 5: Precise Output Format
add_output_format() {
    local prompt="$1"
    local format_section=""
    
    if [[ "$prompt" =~ (code|function|component|implement) ]]; then
        format_section="
# Expected Output Format

Provide your response in this structure:

1. **Brief Analysis** (2-3 sentences explaining the approach)
2. **Implementation** 
   \`\`\`typescript
   // Your code here with comments
   \`\`\`
3. **Usage Example** (if applicable)
4. **Next Steps** (any follow-up tasks or considerations)

Use file references in format: \`src/components/ComponentName.tsx:42\`

"
    elif [[ "$prompt" =~ (debug|bug|error|fix) ]]; then
        format_section="
# Expected Output Format

Structure your debugging response as:

1. **Problem Diagnosis** 
   - Root cause analysis
   - Affected components

2. **Solution**
   \`\`\`typescript
   // Fixed code with explanatory comments
   \`\`\`

3. **Verification Steps**
   - How to test the fix
   - Expected behavior after fix

"
    fi
    
    echo "$prompt$format_section"
}

# Main optimization engine
optimize_prompt() {
    local original="$1"
    local optimized="$original"
    
    log "🧠 Starting OpenAI framework optimization..."
    
    # Skip optimization for very short or already well-structured prompts
    if [[ ${#original} -lt 20 ]] || [[ "$original" =~ ^#.*Role.*Objective ]]; then
        log "ℹ️ Prompt already optimized or too short, skipping"
        echo "$original"
        return
    fi
    
    # Apply OpenAI framework rules in sequence
    optimized=$(add_clear_role "$optimized")
    optimized=$(add_specific_instructions "$optimized")  
    optimized=$(add_chain_of_thought "$optimized")
    optimized=$(add_examples_and_context "$optimized")
    optimized=$(add_output_format "$optimized")
    
    # Add final instructions for persistence (OpenAI agentic best practice)
    if [[ "$optimized" =~ (implement|build|create|fix|debug) ]]; then
        optimized="$optimized
---
**Important**: Please work through this systematically and don't stop until the task is completely resolved. Use available tools to gather information rather than making assumptions.

*🧠 Optimized using OpenAI Prompting Framework | Enhanced by Claude Code Intelligence*"
    fi
    
    # Save optimization results
    echo "$original" > "$CACHE_DIR/last-original.txt"
    echo "$optimized" > "$CACHE_DIR/last-optimized.txt"
    
    # Calculate improvement metrics
    local original_length=${#original}
    local optimized_length=${#optimized}
    local improvement_ratio=$(( (optimized_length * 100) / original_length ))
    
    log "✅ Optimization complete: ${original_length} → ${optimized_length} chars (${improvement_ratio}% expansion)"
    
    echo "$optimized"
}

# Analyze prompt first
analysis_file=$(analyze_prompt_structure "$ORIGINAL_PROMPT")
log "📊 Prompt analysis saved to: $analysis_file"

# Apply optimization
optimized_result=$(optimize_prompt "$ORIGINAL_PROMPT")

# Output the optimized prompt
echo "$optimized_result"