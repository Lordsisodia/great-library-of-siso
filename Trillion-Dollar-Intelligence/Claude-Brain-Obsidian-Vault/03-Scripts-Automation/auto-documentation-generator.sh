#!/bin/bash

# Auto Documentation Generator - Inspired by IsaacWLloyd/cogent-autodoc
FILE_PATH="$1"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/autodoc.log; }

# Check if file should be documented
should_document() {
    local file="$1"
    
    # Document these file types
    case "$file" in
        *.ts|*.tsx|*.js|*.jsx|*.py|*.go|*.rs|*.java|*.php)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Generate documentation path
get_doc_path() {
    local file="$1"
    local dir=$(dirname "$file")
    local basename=$(basename "$file")
    
    # Create .docs directory in same location
    local doc_dir="$dir/.docs"
    mkdir -p "$doc_dir"
    
    echo "$doc_dir/${basename}.md"
}

# Extract code context for analysis
extract_code_context() {
    local file="$1"
    
    cat << EOF
# Code Analysis for $(basename "$file")

## File Overview
**Location**: $file
**Type**: $(file "$file" | cut -d: -f2-)
**Lines**: $(wc -l < "$file" 2>/dev/null || echo "unknown")
**Last Modified**: $(stat -f %Sm "$file" 2>/dev/null || stat -c %y "$file" 2>/dev/null || echo "unknown")

## Code Content
\`\`\`$(get_file_extension "$file")
$(head -100 "$file" 2>/dev/null || echo "Unable to read file")
\`\`\`

## Analysis Request
Please analyze this code and provide:
1. **Purpose & Overview** - What this file does
2. **Key Functions/Classes** - Main components and their roles  
3. **Dependencies** - Imports and external dependencies
4. **Usage Examples** - How to use this code
5. **API Documentation** - Public interfaces and parameters
6. **Error Handling** - How errors are managed
7. **Performance Notes** - Any performance considerations
8. **Maintenance Notes** - Important implementation details

Format as clear, professional documentation.
EOF
}

# Get file extension for syntax highlighting
get_file_extension() {
    local file="$1"
    echo "${file##*.}"
}

# Generate smart documentation
generate_documentation() {
    local file="$1"
    local doc_path="$2"
    
    log "📝 Generating documentation for: $file"
    
    # Create context for AI analysis
    local context_file="/tmp/claude-autodoc-$(date +%s).md"
    extract_code_context "$file" > "$context_file"
    
    # Check if Claude Code is available for analysis
    if command -v claude >/dev/null; then
        log "🤖 Using Claude for documentation analysis"
        
        # Use Claude to analyze and document
        claude -p "$(cat "$context_file")" > "$doc_path" 2>/dev/null || {
            log "⚠️ Claude analysis failed, generating basic documentation"
            generate_basic_documentation "$file" "$doc_path"
        }
    else
        log "📋 Generating basic documentation (Claude not available)"
        generate_basic_documentation "$file" "$doc_path"
    fi
    
    # Cleanup
    rm -f "$context_file"
}

# Generate basic documentation without AI
generate_basic_documentation() {
    local file="$1"
    local doc_path="$2"
    
    cat > "$doc_path" << EOF
# Documentation: $(basename "$file")

**Generated**: $TIMESTAMP  
**File**: $file  
**Type**: $(get_file_extension "$file")  

## Overview
This file contains $(wc -l < "$file" 2>/dev/null || echo "unknown") lines of $(get_file_extension "$file") code.

## Functions and Exports
$(grep -n "function\|export\|class\|def\|func" "$file" 2>/dev/null | head -10 || echo "No functions detected")

## Imports and Dependencies
$(grep -n "import\|require\|from\|#include" "$file" 2>/dev/null | head -10 || echo "No imports detected")

## TODO Items
$(grep -n "TODO\|FIXME\|XXX" "$file" 2>/dev/null || echo "No TODO items found")

## File Structure
\`\`\`
$(head -20 "$file" 2>/dev/null | nl || echo "Unable to read file")
...
\`\`\`

---
*Auto-generated documentation - Update manually as needed*
EOF
}

# Update documentation index
update_doc_index() {
    local doc_path="$1"
    local file="$2"
    local project_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    local index_file="$project_root/.docs/README.md"
    
    # Create docs index if it doesn't exist
    if [[ ! -f "$index_file" ]]; then
        mkdir -p "$(dirname "$index_file")"
        cat > "$index_file" << EOF
# Project Documentation Index

Auto-generated documentation for project files.

## Documented Files
EOF
    fi
    
    # Add entry if not already present
    local relative_path=$(realpath --relative-to="$project_root" "$file" 2>/dev/null || echo "$file")
    local relative_doc=$(realpath --relative-to="$(dirname "$index_file")" "$doc_path" 2>/dev/null || echo "$doc_path")
    
    if ! grep -q "$relative_path" "$index_file"; then
        echo "- [$relative_path]($relative_doc)" >> "$index_file"
        log "📚 Added to documentation index: $relative_path"
    fi
}

# Check for documentation updates needed
check_for_updates() {
    local file="$1"
    local doc_path="$2"
    
    if [[ -f "$doc_path" ]]; then
        local file_time=$(stat -f %m "$file" 2>/dev/null || stat -c %Y "$file" 2>/dev/null || echo "0")
        local doc_time=$(stat -f %m "$doc_path" 2>/dev/null || stat -c %Y "$doc_path" 2>/dev/null || echo "0")
        
        if [[ $file_time -gt $doc_time ]]; then
            log "🔄 Documentation outdated, regenerating: $file"
            return 0  # Needs update
        else
            log "✅ Documentation up to date: $file"
            return 1  # No update needed
        fi
    else
        return 0  # Documentation doesn't exist
    fi
}

# Main execution
main() {
    if [[ ! -f "$FILE_PATH" ]]; then
        log "❌ File not found: $FILE_PATH"
        exit 1
    fi
    
    if ! should_document "$FILE_PATH"; then
        log "⏭️ Skipping documentation (file type not supported): $FILE_PATH"
        exit 0
    fi
    
    local doc_path=$(get_doc_path "$FILE_PATH")
    
    if check_for_updates "$FILE_PATH" "$doc_path"; then
        generate_documentation "$FILE_PATH" "$doc_path"
        update_doc_index "$doc_path" "$FILE_PATH"
        log "✅ Documentation generated: $doc_path"
    fi
}

# Initialize
mkdir -p ~/.claude/logs

# Execute
main