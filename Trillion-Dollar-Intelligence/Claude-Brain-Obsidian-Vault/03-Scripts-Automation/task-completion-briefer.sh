#!/bin/bash

# Task Completion Briefer - Provides brief summaries of completed tasks
TOOL_NAME="$1"
FILE_PATHS="$2"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Generate brief task summary
generate_task_brief() {
    local tool="$1"
    local files="$2"
    
    case "$tool" in
        "Edit")
            if [[ -n "$files" ]]; then
                local file_count=$(echo "$files" | wc -w)
                if [[ $file_count -eq 1 ]]; then
                    echo "📝 EDITED: $(basename "$files")"
                else
                    echo "📝 EDITED: $file_count files"
                fi
            else
                echo "📝 FILE EDITED"
            fi
            ;;
        "Write")
            if [[ -n "$files" ]]; then
                echo "✏️ CREATED: $(basename "$files")"
            else
                echo "✏️ FILE CREATED"
            fi
            ;;
        "MultiEdit")
            local file_count=$(echo "$files" | wc -w)
            echo "📄 MULTI-EDIT: $file_count files modified"
            ;;
        "Read")
            if [[ -n "$files" ]]; then
                echo "👁️ READ: $(basename "$files")"
            else
                echo "👁️ FILE READ"
            fi
            ;;
        "Bash")
            echo "💻 COMMAND EXECUTED"
            ;;
        "WebFetch")
            echo "🌐 WEB CONTENT FETCHED"
            ;;
        "WebSearch")
            echo "🔍 WEB SEARCH PERFORMED"
            ;;
        "Grep")
            echo "🔎 CODE SEARCH COMPLETED"
            ;;
        "Glob")
            echo "📁 FILE PATTERN SEARCH"
            ;;
        "TodoWrite")
            echo "✅ TODO LIST UPDATED"
            ;;
        "mcp__supabase__execute_sql")
            echo "🗄️ DATABASE QUERY EXECUTED"
            ;;
        "mcp__supabase__apply_migration")
            echo "🗄️ DATABASE MIGRATION APPLIED"
            ;;
        "mcp__github__"*)
            echo "🐙 GITHUB OPERATION COMPLETED"
            ;;
        "mcp__notion__"*)
            echo "📝 NOTION OPERATION COMPLETED"
            ;;
        *)
            echo "🔧 TASK COMPLETED: $tool"
            ;;
    esac
}

# Add context based on file types
add_file_context() {
    local files="$1"
    local context=""
    
    if [[ -n "$files" ]]; then
        case "$files" in
            *.ts|*.tsx)
                context=" (TypeScript)"
                ;;
            *.js|*.jsx)
                context=" (JavaScript)"
                ;;
            *.py)
                context=" (Python)"
                ;;
            *.md)
                context=" (Markdown)"
                ;;
            *.json)
                context=" (JSON)"
                ;;
            *.css|*.scss)
                context=" (Styles)"
                ;;
            *test*|*spec*)
                context=" (Tests)"
                ;;
            *package.json*)
                context=" (Dependencies)"
                ;;
        esac
    fi
    
    echo "$context"
}

# Smart project context
get_project_context() {
    local files="$1"
    
    if [[ "$files" =~ SISO ]]; then
        echo " [SISO]"
    elif [[ -f "package.json" ]]; then
        local project_name=$(grep '"name"' package.json 2>/dev/null | cut -d'"' -f4 | head -1)
        if [[ -n "$project_name" ]]; then
            echo " [$project_name]"
        fi
    fi
}

# Main brief generation
main() {
    local brief=$(generate_task_brief "$TOOL_NAME" "$FILE_PATHS")
    local context=$(add_file_context "$FILE_PATHS")
    local project=$(get_project_context "$FILE_PATHS")
    
    # Output the brief
    echo "${brief}${context}${project}"
    
    # Log for analytics
    echo "[$TIMESTAMP] BRIEF: ${brief}${context}${project}" >> ~/.claude/logs/task-briefs.log
}

# Initialize
mkdir -p ~/.claude/logs

# Execute
main