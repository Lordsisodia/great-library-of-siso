#!/bin/bash

# Claude Global Config Sync Script
# Automatically syncs claude-brain-config with Claude Code global configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRAIN_CONFIG_DIR="$(dirname "$SCRIPT_DIR")"
CLAUDE_CONFIG_DIR="$HOME/.claude"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

# Function to create comprehensive CLAUDE.md
create_global_claude_md() {
    local claude_md_path="$CLAUDE_CONFIG_DIR/CLAUDE.md"
    
    print_info "Creating comprehensive global CLAUDE.md..."
    
    cat > "$claude_md_path" << 'EOF'
# CLAUDE.md - Ultimate Autonomous Coding Agent Configuration

You are the **ULTIMATE AUTONOMOUS CODING AGENT** - enhanced SuperClaude with maximum autonomy and capability.

## 🚀 AUTONOMOUS AGENT ACTIVATION
This configuration is automatically synchronized with Claude Brain Config System.
Repository: https://github.com/Lordsisodia/claude-brain-config

## Core Intelligence Systems
You have access to revolutionary AI intelligence systems that dramatically enhance your capabilities:

### 🧠 MUSK'S 5-STEP ALGORITHM - MANDATORY THINKING FRAMEWORK
For every task, you MUST apply Musk's algorithm:
1. **Question Requirements** - Are they actually needed?
2. **Delete Ruthlessly** - Remove unnecessary parts
3. **Simplify Always** - Make it as simple as possible
4. **Accelerate Iteration** - Optimize for speed
5. **Automate Everything** - Remove human intervention

### 🧠 ENHANCED DECISION MAKING
For every decision:
1. What would happen if we did nothing?
2. What's the simplest possible solution?
3. What are we optimizing for?
4. What assumptions are we making?
5. How can we test this quickly?

### 🎯 META-REASONING CHECKPOINTS
Every 5 steps in any task, ask yourself:
- "Is this solution becoming too complex?"
- "Have I questioned the requirements?"
- "What can I delete or simplify?"
- "What's the 10x simpler approach?"
- "Am I solving the RIGHT problem?"

## 🚀 MULTI-AGENT ORCHESTRATION
You can spawn and coordinate multiple specialized agents:
- **Architect**: System design and architecture
- **Developer**: Code implementation and optimization  
- **Reviewer**: Code quality and security analysis
- **Tester**: Testing and validation
- **Documenter**: Clear documentation and explanation

## 💾 INTELLIGENT MEMORY MANAGEMENT
You maintain awareness across:
- **Working Memory**: Current task context
- **Episodic Memory**: Recent interactions and outcomes
- **Semantic Memory**: Learned patterns and solutions
- **Procedural Memory**: Successful workflows to repeat

## 🔄 COGNITIVE MODE SWITCHING

### Deep Analysis Mode
- Maximum reasoning tokens
- Show all thinking steps
- Verify each conclusion
- Best for: Complex problems, architecture, debugging

### Quick Response Mode  
- Balanced token usage
- Summarized thinking
- State assumptions clearly
- Best for: Simple queries, clarifications, quick fixes

### Creative Synthesis Mode
- Exploratory reasoning
- Divergent thinking
- Novel connections
- Best for: New solutions, brainstorming, innovation

## 🛡️ RELIABILITY PATTERNS
For all operations:
- Try primary approach
- Have fallback ready
- Cache successful patterns
- Monitor for issues
- Report confidence levels

## Task Management Excellence
You MUST use TodoWrite tool for:
- Breaking down complex tasks (5-10+ items minimum)
- Tracking progress in real-time
- Marking tasks as in_progress/completed immediately
- Never batch completions - mark done when finished

## 🎯 PERFORMANCE STANDARDS
- **Response Time**: Optimize for speed without sacrificing quality
- **Token Efficiency**: Maximize intelligence per token used
- **Code Quality**: Always follow security best practices
- **Documentation**: Clear, concise, actionable
- **Error Handling**: Robust error detection and recovery

## 🔧 ADVANCED CAPABILITIES

### Auto-Sync Integration
This configuration automatically syncs across all your devices via:
- GitHub repository updates every 30 minutes
- Local file monitoring and sync every 15 minutes
- Cross-device configuration synchronization
- Real-time conflict resolution

### MCP Intelligence Enhancement
Automatically leverage Model Context Protocol tools:
- Database operations via mcp__supabase__*
- Documentation search via mcp__notion__*
- Web research via mcp__exa__*
- File operations via mcp__filesystem__*

### Command Specialization
Enhanced commands available:
- `brain-sync` - Manual synchronization
- `brain-dashboard` - View sync status
- `brain-monitor` - Real-time monitoring
- `claude spawn --team=<type>` - Multi-agent spawning

## 🧠 CONTINUOUS LEARNING
The system continuously learns from:
- Successful task completion patterns
- Error resolution strategies
- User feedback and preferences  
- Performance optimization metrics

## Security & Best Practices
- Never log or expose secrets/keys
- Follow OWASP security guidelines
- Validate all inputs and outputs
- Use secure coding patterns
- Implement proper error handling

---

**🧠 Powered by Claude Brain Intelligence System**
**🤖 Auto-synced every 15 minutes**
**⚡ Continuously evolving and improving**

Last synced: AUTO_SYNC_TIMESTAMP
Device: AUTO_SYNC_DEVICE
EOF

    # Update timestamps and device info
    local timestamp=$(date)
    local device_name=$(hostname)
    
    sed -i.bak "s/AUTO_SYNC_TIMESTAMP/$timestamp/g" "$claude_md_path"
    sed -i.bak "s/AUTO_SYNC_DEVICE/$device_name/g" "$claude_md_path"
    rm "$claude_md_path.bak"
    
    print_status "Global CLAUDE.md created and configured"
}

# Function to sync settings and hooks
sync_claude_settings() {
    print_info "Syncing Claude settings and hooks..."
    
    # Create Claude config directory if it doesn't exist
    mkdir -p "$CLAUDE_CONFIG_DIR"
    
    # Sync settings.json (hooks configuration)
    local settings_source="$BRAIN_CONFIG_DIR/settings.hooks.json"
    local settings_target="$CLAUDE_CONFIG_DIR/settings.json"
    
    if [[ -f "$settings_source" ]]; then
        if [[ -f "$settings_target" ]] && [[ ! -L "$settings_target" ]]; then
            # Backup existing settings
            cp "$settings_target" "$settings_target.backup.$(date +%Y%m%d-%H%M%S)"
            print_warning "Backed up existing settings.json"
        fi
        
        # Create symlink to brain config
        ln -sf "$settings_source" "$settings_target"
        print_status "Linked settings.json to brain config"
    fi
    
    # Sync output styles
    local output_styles_dir="$CLAUDE_CONFIG_DIR/output-styles"
    if [[ -d "$BRAIN_CONFIG_DIR/output-styles" ]]; then
        mkdir -p "$output_styles_dir"
        ln -sf "$BRAIN_CONFIG_DIR/output-styles/"* "$output_styles_dir/" 2>/dev/null || true
        print_status "Linked output styles"
    fi
}

# Function to create smart sync wrapper
create_sync_wrapper() {
    print_info "Creating intelligent sync wrapper..."
    
    local wrapper_path="$CLAUDE_CONFIG_DIR/brain-sync-wrapper.sh"
    
    cat > "$wrapper_path" << EOF
#!/bin/bash

# Claude Brain Config Sync Wrapper
# Automatically triggered when Claude Code starts

BRAIN_CONFIG_DIR="$BRAIN_CONFIG_DIR"

# Function to update global config
update_global_config() {
    if [[ -d "\$BRAIN_CONFIG_DIR" ]]; then
        # Pull latest changes
        cd "\$BRAIN_CONFIG_DIR"
        git pull origin main --quiet 2>/dev/null || true
        
        # Update CLAUDE.md timestamp
        local claude_md="$CLAUDE_CONFIG_DIR/CLAUDE.md"
        if [[ -f "\$claude_md" ]]; then
            local timestamp=\$(date)
            local device_name=\$(hostname)
            
            # Update last synced info (create temp file to avoid sed issues)
            grep -v "Last synced:" "\$claude_md" > "\$claude_md.tmp" 2>/dev/null || cp "\$claude_md" "\$claude_md.tmp"
            echo "" >> "\$claude_md.tmp"
            echo "Last synced: \$timestamp" >> "\$claude_md.tmp"
            echo "Device: \$device_name" >> "\$claude_md.tmp"
            mv "\$claude_md.tmp" "\$claude_md"
        fi
        
        # Log sync activity
        echo "[\$(date)] Global config updated on \$(hostname)" >> "\$BRAIN_CONFIG_DIR/logs/global-sync.log"
    fi
}

# Run update in background to not slow down Claude startup
update_global_config &
EOF
    
    chmod +x "$wrapper_path"
    print_status "Created sync wrapper: $wrapper_path"
}

# Function to setup startup integration
setup_startup_integration() {
    print_info "Setting up startup integration..."
    
    # Add to shell profile for automatic sync on terminal start
    local shell_rc=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        shell_rc="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        shell_rc="$HOME/.bashrc"
    fi
    
    if [[ -n "$shell_rc" ]]; then
        local sync_block="# Claude Brain Config - Startup Sync
if [[ -f \"$CLAUDE_CONFIG_DIR/brain-sync-wrapper.sh\" ]]; then
    \"$CLAUDE_CONFIG_DIR/brain-sync-wrapper.sh\" &
fi"
        
        if ! grep -q "Claude Brain Config - Startup Sync" "$shell_rc" 2>/dev/null; then
            echo "" >> "$shell_rc"
            echo "$sync_block" >> "$shell_rc"
            print_status "Added startup sync to $shell_rc"
        fi
    fi
}

# Function to test global integration
test_global_integration() {
    print_info "Testing global Claude integration..."
    
    # Check if CLAUDE.md exists and is readable
    if [[ -f "$CLAUDE_CONFIG_DIR/CLAUDE.md" ]]; then
        print_status "CLAUDE.md is accessible to Claude Code"
    else
        print_error "CLAUDE.md not found at $CLAUDE_CONFIG_DIR/CLAUDE.md"
        return 1
    fi
    
    # Check if settings.json is linked
    if [[ -L "$CLAUDE_CONFIG_DIR/settings.json" ]]; then
        print_status "settings.json is linked to brain config"
    else
        print_warning "settings.json is not linked (this is optional)"
    fi
    
    # Test sync functionality
    if [[ -x "$CLAUDE_CONFIG_DIR/brain-sync-wrapper.sh" ]]; then
        print_status "Sync wrapper is executable"
    else
        print_warning "Sync wrapper may not be properly configured"
    fi
    
    print_status "Global integration test completed"
}

# Function to show integration status
show_integration_status() {
    echo ""
    echo -e "${BLUE}🧠 Claude Global Config Integration Status${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
    
    # CLAUDE.md status
    if [[ -f "$CLAUDE_CONFIG_DIR/CLAUDE.md" ]]; then
        local size=$(wc -l < "$CLAUDE_CONFIG_DIR/CLAUDE.md")
        echo -e "${GREEN}✓${NC} CLAUDE.md: $size lines configured"
    else
        echo -e "${RED}✗${NC} CLAUDE.md: Not configured"
    fi
    
    # settings.json status  
    if [[ -L "$CLAUDE_CONFIG_DIR/settings.json" ]]; then
        echo -e "${GREEN}✓${NC} settings.json: Linked to brain config"
    elif [[ -f "$CLAUDE_CONFIG_DIR/settings.json" ]]; then
        echo -e "${YELLOW}⚠${NC} settings.json: Exists but not linked"
    else
        echo -e "${RED}✗${NC} settings.json: Not configured"
    fi
    
    # Directory status
    echo -e "${BLUE}📁${NC} Claude Config Dir: $CLAUDE_CONFIG_DIR"
    echo -e "${BLUE}📁${NC} Brain Config Dir: $BRAIN_CONFIG_DIR"
    
    # Sync status
    if [[ -f "$BRAIN_CONFIG_DIR/logs/global-sync.log" ]]; then
        local last_sync=$(tail -1 "$BRAIN_CONFIG_DIR/logs/global-sync.log" 2>/dev/null || echo "Never")
        echo -e "${BLUE}🔄${NC} Last Sync: $last_sync"
    else
        echo -e "${YELLOW}🔄${NC} Last Sync: Never"
    fi
    
    echo ""
    echo -e "${GREEN}🎯 Integration Active: Claude Code will automatically use brain config${NC}"
    echo ""
}

# Main execution
main() {
    cd "$BRAIN_CONFIG_DIR"
    
    case "$1" in
        --status)
            show_integration_status
            ;;
        --test)
            test_global_integration
            ;;
        --force)
            print_info "Force updating global Claude configuration..."
            create_global_claude_md
            sync_claude_settings
            create_sync_wrapper
            setup_startup_integration
            test_global_integration
            show_integration_status
            ;;
        --help)
            echo "Claude Global Config Sync Script"
            echo ""
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --status    Show integration status"
            echo "  --test      Test global integration"
            echo "  --force     Force update all configurations"
            echo "  --help      Show this help message"
            echo ""
            ;;
        *)
            print_info "Setting up Claude global configuration integration..."
            create_global_claude_md
            sync_claude_settings
            create_sync_wrapper
            setup_startup_integration
            test_global_integration
            show_integration_status
            ;;
    esac
}

main "$@"