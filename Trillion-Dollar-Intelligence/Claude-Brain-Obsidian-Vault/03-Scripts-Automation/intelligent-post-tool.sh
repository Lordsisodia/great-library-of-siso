#!/bin/bash

# 10x Intelligent Post-Tool Processor - Smart automation after any tool use
TOOL_NAME="$1"
FILE_PATHS="$2"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/hooks.log; }

# Smart tool-specific automation
case "$TOOL_NAME" in
    "Edit"|"Write"|"MultiEdit")
        log "📝 File modification detected: $FILE_PATHS"
        
        # Smart project context detection
        for file in $FILE_PATHS; do
            if [[ "$file" == *"SISO"* ]]; then
                log "🏢 SISO project file modified: $file"
                echo "$TIMESTAMP,$file,SISO_EDIT" >> ~/.claude/analytics/siso-activity.csv
            fi
            
            # Auto-suggest related actions
            case "$file" in
                *component*.tsx|*Component*.tsx)
                    log "⚛️ React component edited - consider Storybook update"
                    ;;
                *api*.ts|*API*.ts)
                    log "🌐 API file edited - consider Postman collection update"
                    ;;
                *schema*.ts|*Schema*.ts)
                    log "🗄️ Schema file edited - consider database migration"
                    ;;
                *.md)
                    log "📚 Documentation updated - consider publishing"
                    ;;
            esac
        done
        ;;
        
    "Bash")
        log "💻 Bash command executed"
        
        # Monitor risky commands
        if [[ "$FILE_PATHS" == *"rm"* ]] || [[ "$FILE_PATHS" == *"delete"* ]]; then
            log "⚠️ Destructive command detected - logged for review"
            echo "$TIMESTAMP,DESTRUCTIVE,$FILE_PATHS" >> ~/.claude/logs/risky-commands.log
        fi
        ;;
        
    "mcp__supabase__execute_sql"|"mcp__supabase__apply_migration")
        log "🗄️ Database operation completed"
        echo "$TIMESTAMP,DB_OPERATION,$TOOL_NAME" >> ~/.claude/analytics/database-activity.csv
        
        # Auto-suggest related actions
        log "💡 Consider: backup verification, type regeneration, schema docs update"
        ;;
        
    "mcp__github__*")
        log "🐙 GitHub operation completed"
        echo "$TIMESTAMP,GITHUB_OP,$TOOL_NAME" >> ~/.claude/analytics/github-activity.csv
        ;;
        
    "WebFetch"|"WebSearch")
        log "🌐 Web operation completed"
        # Track research patterns
        echo "$TIMESTAMP,WEB_RESEARCH,$TOOL_NAME" >> ~/.claude/analytics/research-activity.csv
        ;;
esac

# Session context awareness
if [[ -f ~/.claude/session-context ]]; then
    context=$(cat ~/.claude/session-context)
    
    case "$context" in
        *"DEBUG_MODE=true"*)
            log "🐛 Debug mode active - enhanced error tracking enabled"
            ;;
        *"FEATURE_MODE=true"*)
            log "✨ Feature mode active - progress tracking enabled"
            ;;
        *"REFACTOR_MODE=true"*)
            log "🔧 Refactor mode active - quality metrics enabled"
            ;;
    esac
fi

# Smart suggestions based on activity patterns
suggest_next_actions() {
    # Check recent activity for patterns
    if [[ -f ~/.claude/analytics/siso-activity.csv ]]; then
        recent_edits=$(tail -10 ~/.claude/analytics/siso-activity.csv | wc -l)
        
        if [[ $recent_edits -ge 5 ]]; then
            log "🎯 High activity detected - consider running full test suite"
        fi
    fi
}

# Initialize analytics
mkdir -p ~/.claude/analytics

# Run smart suggestions
suggest_next_actions

log "🧠 Intelligent post-tool processing complete"