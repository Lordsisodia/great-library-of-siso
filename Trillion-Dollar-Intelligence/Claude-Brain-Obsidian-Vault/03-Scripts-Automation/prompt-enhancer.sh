#!/bin/bash

# 100x Prompt Enhancement Engine
# Automatically analyzes prompts and adds relevant context, code references, and smart optimizations

PROMPT="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
PROJECT_ROOT=$(pwd)

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/prompt-enhancer.log; }
error() { echo "[$TIMESTAMP] ERROR: $1" >> ~/.claude/logs/prompt-enhancer.log; }

# Create necessary directories
mkdir -p ~/.claude/logs ~/.claude/cache ~/.claude/analytics

# Enhanced prompt analysis with context discovery
analyze_and_enhance_prompt() {
    local original_prompt="$1"
    local enhanced_prompt="$original_prompt"
    local context_additions=""
    
    log "🔍 Analyzing prompt: ${original_prompt:0:100}..."
    
    # 1. DETECT INTENT AND ADD SMART CONTEXT
    detect_intent_and_add_context() {
        # Code-related prompts
        if [[ "$original_prompt" =~ (fix|bug|error|debug|issue) ]]; then
            context_additions="$context_additions

🐛 **DEBUG MODE ACTIVATED**
- Check recent git changes: \`git log --oneline -10\`
- Look for error patterns in logs
- Consider running tests first to isolate the issue"
            
            # Find recent error-prone files
            if command -v git >/dev/null 2>&1; then
                local recent_files=$(git diff --name-only HEAD~5..HEAD 2>/dev/null | head -5)
                if [[ -n "$recent_files" ]]; then
                    context_additions="$context_additions
- **Recently changed files (potential bug sources):**
$(echo "$recent_files" | sed 's/^/  - /')"
                fi
            fi
        fi
        
        # Feature development
        if [[ "$original_prompt" =~ (add|create|implement|feature|new) ]]; then
            context_additions="$context_additions

✨ **FEATURE DEVELOPMENT MODE**
- Follow existing code patterns in the codebase
- Check for similar implementations to reuse patterns
- Consider test coverage for new functionality"
            
            # Find similar components/files
            local key_words=$(echo "$original_prompt" | grep -oE '\b[A-Z][a-z]*[A-Z][a-z]*\b|\b[a-z]+[A-Z][a-z]*\b' | head -3)
            if [[ -n "$key_words" ]]; then
                context_additions="$context_additions
- **Search for similar patterns:** Look for files containing: $key_words"
            fi
        fi
        
        # UI/Component work
        if [[ "$original_prompt" =~ (component|ui|interface|design|style|css) ]]; then
            context_additions="$context_additions

🎨 **UI/COMPONENT MODE**
- Check existing design system components first
- Follow established styling patterns (Tailwind classes, etc.)
- Consider responsive design and accessibility"
            
            # Find existing components
            if [[ -d "src/components" ]]; then
                local existing_components=$(find src/components -name "*.tsx" -o -name "*.jsx" 2>/dev/null | head -5)
                if [[ -n "$existing_components" ]]; then
                    context_additions="$context_additions
- **Existing component references:**
$(echo "$existing_components" | sed 's/^/  - /')"
                fi
            fi
        fi
        
        # Database/API work
        if [[ "$original_prompt" =~ (database|api|endpoint|query|supabase|sql) ]]; then
            context_additions="$context_additions

🗄️ **DATABASE/API MODE**
- Check existing schemas and types
- Follow established API patterns
- Consider error handling and validation"
            
            # Find type definitions
            if [[ -f "src/integrations/supabase/types.ts" ]]; then
                context_additions="$context_additions
- **Database types available:** src/integrations/supabase/types.ts"
            fi
        fi
    }
    
    # 2. DISCOVER RELEVANT FILES
    discover_relevant_files() {
        local search_terms=$(echo "$original_prompt" | tr '[:upper:]' '[:lower:]' | grep -oE '\b[a-z]{3,}\b' | grep -v -E '^(the|and|for|with|from|this|that|can|you|help|me|please|i|we|my|our)$' | head -5)
        
        if [[ -n "$search_terms" ]]; then
            log "🔍 Searching for relevant files with terms: $search_terms"
            
            local relevant_files=""
            for term in $search_terms; do
                # Search in common directories
                local files=$(find . -type f \( -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" -o -name "*.py" -o -name "*.md" \) \
                    -not -path "./node_modules/*" -not -path "./.git/*" -not -path "./dist/*" -not -path "./build/*" \
                    2>/dev/null | xargs grep -l "$term" 2>/dev/null | head -3)
                
                if [[ -n "$files" ]]; then
                    relevant_files="$relevant_files$files\n"
                fi
            done
            
            if [[ -n "$relevant_files" ]]; then
                context_additions="$context_additions

📁 **RELEVANT FILES FOUND:**
$(echo -e "$relevant_files" | sort -u | head -8 | sed 's/^/- /')"
            fi
        fi
    }
    
    # 3. ADD PROJECT-SPECIFIC CONTEXT
    add_project_context() {
        # SISO Ecosystem specific
        if [[ "$PROJECT_ROOT" =~ SISO|siso ]] || [[ "$original_prompt" =~ SISO|siso ]]; then
            context_additions="$context_additions

🏢 **SISO ECOSYSTEM CONTEXT**
- Follow SISO brand guidelines (orange/yellow theme)
- Check CLAUDE.md for project-specific patterns
- Run quality checks: \`npm run lint && npm run build\`"
            
            # Detect which SISO project
            if [[ -d "SISO-CLIENT-BASE" ]]; then
                context_additions="$context_additions
- **Main Platform:** SISO-CLIENT-BASE/siso-agency-onboarding-app-main-dev/"
            fi
            if [[ -d "SISO-DEV-TOOLS" ]]; then
                context_additions="$context_additions
- **Dev Tools:** SISO-DEV-TOOLS/ (Claudia Fresh, ClaudeCodeUI)"
            fi
        fi
        
        # React/TypeScript project
        if [[ -f "package.json" ]] && grep -q "react\|typescript" package.json 2>/dev/null; then
            context_additions="$context_additions

⚛️ **REACT/TYPESCRIPT PROJECT**
- Use existing component patterns
- Follow TypeScript best practices
- Check for existing hooks and utilities"
        fi
    }
    
    # 4. ENHANCE PROMPT WITH SMART SUGGESTIONS
    add_smart_enhancements() {
        # Make vague requests more specific
        if [[ "$original_prompt" =~ ^(help|can you|please|fix this|make this better)$ ]]; then
            context_additions="$context_additions

💡 **ENHANCED REQUEST NEEDED**
Your prompt could be more specific. Consider adding:
- What specific file or component needs attention?
- What exact behavior do you want?
- What error or issue are you seeing?
- What's the expected outcome?"
        fi
        
        # Add efficiency suggestions
        if [[ "$original_prompt" =~ (search|find|look) ]]; then
            context_additions="$context_additions

🚀 **EFFICIENCY TIP**
Consider using specific file paths or grep patterns for faster results:
- Use \`grep -r \"pattern\" src/\` for code searches
- Reference specific files: \`src/components/ComponentName.tsx\`
- Use glob patterns: \`**/*.tsx\` for file searches"
        fi
    }
    
    # Execute all enhancement functions
    detect_intent_and_add_context
    discover_relevant_files
    add_project_context
    add_smart_enhancements
    
    # 5. CONSTRUCT ENHANCED PROMPT
    if [[ -n "$context_additions" ]]; then
        enhanced_prompt="$original_prompt$context_additions

---
*🤖 Auto-enhanced by Claude Code Prompt Engine - Context added to 100x your outcome*"
        
        log "✅ Prompt enhanced with $(echo "$context_additions" | wc -l) lines of context"
        
        # Cache the enhancement for learning
        echo "$original_prompt" > ~/.claude/cache/last-original-prompt
        echo "$enhanced_prompt" > ~/.claude/cache/last-enhanced-prompt
        
        # Track enhancement metrics
        echo "$(date '+%Y-%m-%d %H:%M:%S'),${#original_prompt},${#enhanced_prompt},$(echo "$context_additions" | wc -l)" >> ~/.claude/analytics/enhancement-metrics.csv
    else
        log "ℹ️ No enhancements needed for this prompt"
    fi
    
    # Output the enhanced prompt
    echo "$enhanced_prompt"
}

# Main execution
main() {
    if [[ -z "$PROMPT" ]]; then
        error "No prompt provided"
        exit 1
    fi
    
    log "🚀 Starting prompt enhancement process"
    
    # Analyze and enhance the prompt
    enhanced_result=$(analyze_and_enhance_prompt "$PROMPT")
    
    # Output the result (this will be used by Claude Code)
    echo "$enhanced_result"
    
    log "🎯 Prompt enhancement complete"
}

# Run the main function
main "$@"