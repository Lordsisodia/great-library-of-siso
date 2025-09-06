#!/bin/bash

# 🏷️ Claude Brain File Categorization System
# Automatically categorizes and tags files for better navigation

BRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAGS_FILE="$BRAIN_DIR/FILE-CATEGORIES.json"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 Claude Brain File Categorization System${NC}"
echo "Analyzing $BRAIN_DIR"
echo

# Initialize categories
declare -A categories=(
    ["intelligence"]=""
    ["automation"]=""
    ["agents"]=""
    ["documentation"]=""
    ["performance"]=""
    ["learning"]=""
    ["monitoring"]=""
    ["integration"]=""
    ["templates"]=""
    ["data"]=""
)

# Categorization patterns
categorize_file() {
    local file="$1"
    local basename=$(basename "$file")
    local dirname=$(dirname "$file" | sed "s|$BRAIN_DIR/||")
    local tags=()
    
    # By directory
    case "$dirname" in
        "shared"*)
            tags+=("intelligence")
            if [[ "$basename" =~ intelligence ]]; then
                tags+=("core-intelligence")
            fi
            ;;
        "agents"*)
            tags+=("agents")
            if [[ "$basename" =~ orchestrator ]]; then
                tags+=("coordination")
            elif [[ "$basename" =~ enhanced ]]; then
                tags+=("enhanced")
            fi
            ;;
        "scripts"*)
            tags+=("automation")
            if [[ "$basename" =~ auto ]]; then
                tags+=("auto-system")
            elif [[ "$basename" =~ optimizer ]]; then
                tags+=("optimization")
            elif [[ "$basename" =~ monitor ]]; then
                tags+=("monitoring")
            fi
            ;;
        "analytics"*)
            tags+=("data" "performance")
            ;;
        "logs"*)
            tags+=("monitoring" "data")
            ;;
        "todos"*)
            tags+=("data" "task-management")
            ;;
        "templates"*)
            tags+=("templates")
            ;;
    esac
    
    # By filename patterns
    if [[ "$basename" =~ INTELLIGENCE|intelligence ]]; then
        tags+=("intelligence")
    fi
    
    if [[ "$basename" =~ GUIDE|README|INDEX ]]; then
        tags+=("documentation")
    fi
    
    if [[ "$basename" =~ auto|Auto|AUTO ]]; then
        tags+=("automation")
    fi
    
    if [[ "$basename" =~ optimizer|performance|metrics ]]; then
        tags+=("performance")
    fi
    
    if [[ "$basename" =~ learning|memory|adaptive ]]; then
        tags+=("learning")
    fi
    
    if [[ "$basename" =~ mcp|MCP|integration ]]; then
        tags+=("integration")
    fi
    
    if [[ "$basename" =~ monitor|log|analytics ]]; then
        tags+=("monitoring")
    fi
    
    # Remove duplicates
    local unique_tags=($(printf "%s\n" "${tags[@]}" | sort -u))
    echo "${unique_tags[@]}"
}

# Generate file catalog
generate_catalog() {
    echo -e "${YELLOW}📊 Generating file catalog...${NC}"
    
    local temp_catalog=$(mktemp)
    echo "{" > "$temp_catalog"
    echo "  \"generated\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$temp_catalog"
    echo "  \"total_files\": $(find "$BRAIN_DIR" -type f | wc -l)," >> "$temp_catalog"
    echo "  \"categories\": {" >> "$temp_catalog"
    
    # Process each file
    local file_count=0
    while IFS= read -r -d '' file; do
        local rel_path=$(echo "$file" | sed "s|$BRAIN_DIR/||")
        local tags=($(categorize_file "$file"))
        local tags_json=$(printf '"%s",' "${tags[@]}" | sed 's/,$//')
        
        if [ ${#tags[@]} -gt 0 ]; then
            if [ $file_count -gt 0 ]; then
                echo "," >> "$temp_catalog"
            fi
            echo -n "    \"$rel_path\": [$tags_json]" >> "$temp_catalog"
            ((file_count++))
        fi
    done < <(find "$BRAIN_DIR" -type f \( -name "*.md" -o -name "*.yml" -o -name "*.sh" -o -name "*.py" -o -name "*.json" \) -print0)
    
    echo "" >> "$temp_catalog"
    echo "  }," >> "$temp_catalog"
    
    # Generate category summaries
    echo "  \"summary\": {" >> "$temp_catalog"
    
    # Count files by category
    local intelligence_count=$(grep -o '"intelligence"' "$temp_catalog" | wc -l)
    local automation_count=$(grep -o '"automation"' "$temp_catalog" | wc -l)
    local agents_count=$(grep -o '"agents"' "$temp_catalog" | wc -l)
    local documentation_count=$(grep -o '"documentation"' "$temp_catalog" | wc -l)
    local performance_count=$(grep -o '"performance"' "$temp_catalog" | wc -l)
    local monitoring_count=$(grep -o '"monitoring"' "$temp_catalog" | wc -l)
    
    echo "    \"intelligence_files\": $intelligence_count," >> "$temp_catalog"
    echo "    \"automation_files\": $automation_count," >> "$temp_catalog"
    echo "    \"agent_files\": $agents_count," >> "$temp_catalog"
    echo "    \"documentation_files\": $documentation_count," >> "$temp_catalog"
    echo "    \"performance_files\": $performance_count," >> "$temp_catalog"
    echo "    \"monitoring_files\": $monitoring_count," >> "$temp_catalog"
    echo "    \"categorized_files\": $file_count" >> "$temp_catalog"
    echo "  }" >> "$temp_catalog"
    echo "}" >> "$temp_catalog"
    
    mv "$temp_catalog" "$TAGS_FILE"
    echo -e "${GREEN}✅ Catalog generated: $TAGS_FILE${NC}"
}

# Search functions
search_by_tag() {
    local tag="$1"
    if [ ! -f "$TAGS_FILE" ]; then
        echo -e "${RED}❌ Catalog not found. Run with --generate first.${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}🔍 Files tagged with: $tag${NC}"
    jq -r ".categories | to_entries[] | select(.value[] == \"$tag\") | .key" "$TAGS_FILE"
}

show_categories() {
    if [ ! -f "$TAGS_FILE" ]; then
        echo -e "${RED}❌ Catalog not found. Run with --generate first.${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}📋 Available categories:${NC}"
    jq -r '.categories | [.[] | .[]] | unique | .[]' "$TAGS_FILE" | sort
    echo
    echo -e "${BLUE}📊 Category summary:${NC}"
    jq -r '.summary | to_entries[] | "\(.key): \(.value)"' "$TAGS_FILE"
}

show_file_info() {
    local file="$1"
    if [ ! -f "$TAGS_FILE" ]; then
        echo -e "${RED}❌ Catalog not found. Run with --generate first.${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}🏷️  Tags for: $file${NC}"
    jq -r ".categories[\"$file\"] // [] | .[]" "$TAGS_FILE"
}

# Usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --generate           Generate file categorization catalog"
    echo "  --search <tag>       Find files with specific tag"
    echo "  --categories         Show all available categories"
    echo "  --info <file>        Show tags for specific file"
    echo "  --help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --generate"
    echo "  $0 --search intelligence"
    echo "  $0 --search automation"
    echo "  $0 --info shared/superclaude-core.yml"
    echo "  $0 --categories"
}

# Main execution
case "${1:-}" in
    "--generate")
        generate_catalog
        ;;
    "--search")
        if [ -z "${2:-}" ]; then
            echo -e "${RED}❌ Please specify a tag to search for${NC}"
            exit 1
        fi
        search_by_tag "$2"
        ;;
    "--categories")
        show_categories
        ;;
    "--info")
        if [ -z "${2:-}" ]; then
            echo -e "${RED}❌ Please specify a file path${NC}"
            exit 1
        fi
        show_file_info "$2"
        ;;
    "--help"|"-h"|"")
        show_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        show_usage
        exit 1
        ;;
esac