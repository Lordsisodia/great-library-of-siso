#!/bin/bash

# SISO Ecosystem - Post Tool Execution Hook
# Logs tool usage for analytics and automated responses

TOOL_NAME="$1"
FILE_PATHS="$2"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Ensure logs directory exists
mkdir -p ~/.claude/logs

# Log tool usage
echo "[$TIMESTAMP] TOOL: $TOOL_NAME | FILES: $FILE_PATHS" >> ~/.claude/logs/hooks.log

# Special handling for specific tools
case "$TOOL_NAME" in
    "Edit"|"Write"|"MultiEdit")
        # For file operations, check if we should run additional validations
        if [[ "$FILE_PATHS" == *".ts"* ]] || [[ "$FILE_PATHS" == *".tsx"* ]]; then
            echo "[$TIMESTAMP] TypeScript file modified, consider type checking" >> ~/.claude/logs/hooks.log
        fi
        ;;
    "Bash")
        # Log bash commands for security monitoring
        echo "[$TIMESTAMP] Bash command executed" >> ~/.claude/logs/bash-commands.log
        ;;
    "mcp__supabase__execute_sql"|"mcp__supabase__apply_migration")
        # Log database operations
        echo "[$TIMESTAMP] Database operation via MCP" >> ~/.claude/logs/database-ops.log
        ;;
esac

exit 0