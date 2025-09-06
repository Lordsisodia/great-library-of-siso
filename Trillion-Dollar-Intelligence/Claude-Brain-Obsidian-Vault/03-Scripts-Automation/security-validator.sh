#!/bin/bash

# Enhanced Security Validator - Community patterns from disler/claude-code-hooks-mastery
COMMAND="$1"
FILE_PATHS="$2"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/security.log; }

# Dangerous command patterns from community research
DANGEROUS_PATTERNS=(
    "rm -rf /"
    "sudo rm -rf"
    "format.*"
    "mkfs"
    "dd if=/dev/zero"
    "> /dev/sda"
    "chmod 777"
    "curl.*| sh"
    "wget.*| sh"
    ":(){ :|:& };:"
)

# Check for dangerous patterns
check_dangerous_commands() {
    local cmd="$1"
    
    for pattern in "${DANGEROUS_PATTERNS[@]}"; do
        if [[ "$cmd" =~ $pattern ]]; then
            log "🚨 BLOCKED: Dangerous command detected: $pattern"
            echo "🛡️ SECURITY BLOCK: Command contains dangerous pattern: $pattern"
            echo "❌ Command blocked for safety"
            exit 2  # Community standard: exit 2 for blocking
        fi
    done
}

# Advanced file validation from community patterns
validate_file_access() {
    local files="$1"
    
    # Block access to sensitive files
    local sensitive_patterns=(
        "/etc/passwd"
        "/etc/shadow"
        "~/.ssh/id_rsa"
        "~/.aws/credentials"
        ".env"
        "secrets.json"
    )
    
    for file in $files; do
        for pattern in "${sensitive_patterns[@]}"; do
            if [[ "$file" =~ $pattern ]]; then
                log "🚨 BLOCKED: Sensitive file access attempt: $file"
                echo "🔒 SECURITY BLOCK: Access to sensitive file blocked: $file"
                exit 2
            fi
        done
    done
}

# Context injection for security warnings (community pattern)
inject_security_context() {
    if [[ "$COMMAND" =~ (database|migration|schema) ]]; then
        echo "🗄️ DATABASE OPERATION DETECTED"
        echo "⚠️ Verify: backup exists, operation is reversible, testing complete"
        echo "💡 Consider: running in staging first"
    fi
    
    if [[ "$COMMAND" =~ (deploy|production|release) ]]; then
        echo "🚀 DEPLOYMENT OPERATION DETECTED"
        echo "⚠️ Verify: tests pass, staging validated, rollback plan ready"
        echo "💡 Consider: blue/green deployment strategy"
    fi
}

# Main validation
log "🔍 Security validation started for: $COMMAND"

check_dangerous_commands "$COMMAND"
validate_file_access "$FILE_PATHS"
inject_security_context

log "✅ Security validation passed"
exit 0