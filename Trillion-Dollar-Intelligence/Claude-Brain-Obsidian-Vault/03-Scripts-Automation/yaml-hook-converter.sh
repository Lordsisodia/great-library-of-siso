#!/bin/bash

# YAML Hook Converter - Inspired by syou6162/cchook
YAML_FILE="$1"
JSON_OUTPUT="$2"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TIMESTAMP] $1" >> ~/.claude/logs/yaml-converter.log; }

# Check dependencies
check_dependencies() {
    if ! command -v python3 >/dev/null; then
        log "❌ Python3 required for YAML processing"
        return 1
    fi
    return 0
}

# Convert YAML to JSON using Python
convert_yaml_to_json() {
    local yaml_file="$1"
    local json_file="$2"
    
    python3 << 'EOF'
import sys
import json
import re

def parse_yaml_line(line):
    """Simple YAML parser for hook configurations"""
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None
    
    if ':' in line:
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        
        # Remove quotes
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        
        return key, value
    return None, None

def convert_yaml_hooks(yaml_content):
    """Convert simplified YAML to Claude Code JSON format"""
    lines = yaml_content.split('\n')
    hooks = []
    current_hook = None
    
    for line in lines:
        key, value = parse_yaml_line(line)
        if not key:
            continue
            
        if key in ['PreToolUse', 'PostToolUse', 'UserPromptSubmit', 'Stop']:
            # New hook section
            if current_hook:
                hooks.append(current_hook)
            current_hook = {
                "matcher": "",
                "hooks": []
            }
            hook_type = key
            
        elif key == 'matcher':
            if current_hook:
                current_hook["matcher"] = value
                
        elif key == 'command':
            if current_hook:
                current_hook["hooks"].append({
                    "type": "command",
                    "command": value
                })
                
        elif key == 'conditions':
            # Handle conditions (simplified)
            if current_hook and current_hook["hooks"]:
                # Add condition info to command
                pass
    
    if current_hook:
        hooks.append(current_hook)
    
    return {"hooks": hooks}

# Read YAML file
try:
    with open(sys.argv[1], 'r') as f:
        yaml_content = f.read()
    
    # Convert to JSON
    json_output = convert_yaml_hooks(yaml_content)
    
    # Write JSON file
    with open(sys.argv[2], 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print("✅ Conversion completed successfully")
    
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    sys.exit(1)
EOF
}

# Generate example YAML configuration
generate_example_yaml() {
    local output_file="$1"
    
    cat > "$output_file" << 'EOF'
# Claude Code Hooks - YAML Configuration
# Simplified syntax for easy hook management

PostToolUse:
  - matcher: "Edit.*\\.(ts|tsx)$"
    conditions:
      - type: file_extension
        value: ".ts"
    actions:
      - type: command
        command: "npx prettier --write {.file_path}"
      - type: command
        command: "npx eslint --fix {.file_path}"

  - matcher: "Write.*\\.test\\."
    actions:
      - type: command
        command: "npm test -- {.file_path} --watchAll=false"

PreToolUse:
  - matcher: "Bash.*git.*commit"
    conditions:
      - type: git_status
        value: "has_changes"
    actions:
      - type: message
        content: "🔍 Pre-commit checks running..."
      - type: command
        command: "npm run lint && npm run build"

UserPromptSubmit:
  - matcher: ".*"
    actions:
      - type: command
        command: "echo 'Session: {.timestamp}' >> ~/.claude/session.log"

Stop:
  - matcher: ".*"
    actions:
      - type: command
        command: "echo 'Session completed at {.timestamp}' >> ~/.claude/session.log"
      - type: notification
        title: "Claude Code"
        message: "Session completed successfully"
EOF
}

# Main execution
main() {
    if ! check_dependencies; then
        exit 1
    fi
    
    if [[ -z "$YAML_FILE" ]]; then
        log "📝 No YAML file specified, generating example"
        local example_file="~/.claude/hooks-example.yaml"
        generate_example_yaml "$example_file"
        echo "📋 Example YAML configuration created: $example_file"
        echo "🔧 Edit the file and run: $0 $example_file ~/.claude/settings.hooks.json"
        exit 0
    fi
    
    if [[ ! -f "$YAML_FILE" ]]; then
        log "❌ YAML file not found: $YAML_FILE"
        exit 1
    fi
    
    local json_output="${JSON_OUTPUT:-${YAML_FILE%.*}.json}"
    
    log "🔄 Converting YAML to JSON: $YAML_FILE → $json_output"
    
    # Perform conversion
    if python3 -c "
import sys
sys.argv = ['', '$YAML_FILE', '$json_output']
$(cat << 'PYTHON_SCRIPT'
import json
import re

def parse_yaml_hooks(yaml_content):
    # Simplified YAML to JSON converter for Claude Code hooks
    lines = yaml_content.split('\n')
    hooks = []
    current_section = None
    current_hook = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.endswith(':') and not line.startswith(' '):
            # Main section (PreToolUse, PostToolUse, etc.)
            if current_hook:
                hooks.append(current_hook)
                current_hook = None
            current_section = line[:-1]
            
        elif line.startswith('- matcher:'):
            # New hook in section
            if current_hook:
                hooks.append(current_hook)
            matcher = line.split(':', 1)[1].strip().strip('\"\'')
            current_hook = {
                'matcher': matcher,
                'hooks': []
            }
            
        elif line.startswith('command:') and current_hook:
            command = line.split(':', 1)[1].strip().strip('\"\'')
            current_hook['hooks'].append({
                'type': 'command',
                'command': command
            })
    
    if current_hook:
        hooks.append(current_hook)
    
    return {'hooks': hooks}

# Main conversion
try:
    with open(sys.argv[1], 'r') as f:
        yaml_content = f.read()
    
    json_output = parse_yaml_hooks(yaml_content)
    
    with open(sys.argv[2], 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print('✅ YAML to JSON conversion completed')
except Exception as e:
    print(f'❌ Conversion failed: {e}')
    sys.exit(1)
PYTHON_SCRIPT
)"; then
        log "✅ Conversion successful: $json_output"
        echo "🎯 Converted: $YAML_FILE → $json_output"
        echo "🔧 To activate: cp '$json_output' ~/.claude/settings.hooks.json"
    else
        log "❌ Conversion failed"
        exit 1
    fi
}

# Initialize
mkdir -p ~/.claude/logs

# Execute
main