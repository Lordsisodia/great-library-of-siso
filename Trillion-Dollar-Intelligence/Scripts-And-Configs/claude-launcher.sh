#!/bin/bash

# Colors for better UI
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      🚀 Claude Ultimate Launcher 🚀      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Choose your mode:${NC}"
echo ""
echo -e "  ${YELLOW}1)${NC} 🖥️  Desktop Power Mode (Claudia GUI)"
echo -e "       Advanced agents, timeline, analytics"
echo ""
echo -e "  ${YELLOW}2)${NC} 📱 Mobile/Web Mode (Claude Code UI)"
echo -e "       Access from phone, tablet, or browser"
echo ""
echo -e "  ${YELLOW}3)${NC} 🚀 Launch Both (Ultimate Mode)"
echo -e "       Desktop + Mobile access"
echo ""
echo -e "  ${YELLOW}4)${NC} 🛑 Stop All Services"
echo ""
echo -e "  ${YELLOW}5)${NC} 📊 Check Status"
echo ""
echo -e "  ${YELLOW}6)${NC} 📱 Get Mobile URL"
echo ""
read -p "Enter choice [1-6]: " choice

# Get local IP
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")

case $choice in
  1)
    echo -e "\n${GREEN}Starting Claudia GUI...${NC}"
    cd ~/DEV/claudia-gui 2>/dev/null || { echo -e "${RED}Claudia GUI not found. Please install first.${NC}"; exit 1; }
    source "$HOME/.cargo/env" 2>/dev/null
    bun run tauri dev > /tmp/claudia-gui.log 2>&1 &
    echo $! > /tmp/claudia-gui.pid
    sleep 3
    echo -e "${GREEN}✅ Claudia GUI launched!${NC}"
    echo -e "   Desktop app should open automatically"
    ;;
    
  2)
    echo -e "\n${GREEN}Starting Claude Code UI...${NC}"
    cd ~/DEV/claudecodeui 2>/dev/null || { echo -e "${RED}Claude Code UI not found. Please install first.${NC}"; exit 1; }
    npm run dev > /tmp/claude-ui.log 2>&1 &
    echo $! > /tmp/claude-ui.pid
    sleep 3
    echo -e "${GREEN}✅ Claude Code UI running!${NC}"
    echo -e "   ${BLUE}Local:${NC} http://localhost:3001"
    echo -e "   ${BLUE}Mobile:${NC} http://${LOCAL_IP}:3001"
    echo -e "\n   ${YELLOW}📱 To add to iPhone home screen:${NC}"
    echo -e "   1. Open Safari on iPhone"
    echo -e "   2. Go to http://${LOCAL_IP}:3001"
    echo -e "   3. Tap Share → Add to Home Screen"
    ;;
    
  3)
    echo -e "\n${GREEN}Starting Ultimate Mode...${NC}"
    
    # Start Claudia GUI
    echo -e "Starting Claudia GUI..."
    cd ~/DEV/claudia-gui 2>/dev/null && {
        source "$HOME/.cargo/env" 2>/dev/null
        bun run tauri dev > /tmp/claudia-gui.log 2>&1 &
        echo $! > /tmp/claudia-gui.pid
    }
    
    # Start Claude Code UI
    echo -e "Starting Claude Code UI..."
    cd ~/DEV/claudecodeui 2>/dev/null && {
        npm run dev > /tmp/claude-ui.log 2>&1 &
        echo $! > /tmp/claude-ui.pid
    }
    
    sleep 5
    echo -e "\n${GREEN}✅ Both apps running!${NC}"
    echo -e "   ${BLUE}Desktop:${NC} Claudia GUI window"
    echo -e "   ${BLUE}Web:${NC} http://localhost:3001"
    echo -e "   ${BLUE}Mobile:${NC} http://${LOCAL_IP}:3001"
    ;;
    
  4)
    echo -e "\n${YELLOW}Stopping all services...${NC}"
    
    # Stop using PID files if they exist
    [ -f /tmp/claudia-gui.pid ] && kill $(cat /tmp/claudia-gui.pid) 2>/dev/null && rm /tmp/claudia-gui.pid
    [ -f /tmp/claude-ui.pid ] && kill $(cat /tmp/claude-ui.pid) 2>/dev/null && rm /tmp/claude-ui.pid
    
    # Also try to kill by name as backup
    pkill -f "tauri dev" 2>/dev/null
    pkill -f "npm run dev" 2>/dev/null
    pkill -f "claudecodeui" 2>/dev/null
    
    echo -e "${GREEN}✅ All services stopped${NC}"
    ;;
    
  5)
    echo -e "\n${BLUE}Service Status:${NC}"
    echo ""
    
    # Check Claudia GUI
    if [ -f /tmp/claudia-gui.pid ] && ps -p $(cat /tmp/claudia-gui.pid) > /dev/null 2>&1; then
        echo -e "  Claudia GUI: ${GREEN}● Running${NC}"
    else
        echo -e "  Claudia GUI: ${RED}○ Stopped${NC}"
    fi
    
    # Check Claude Code UI
    if [ -f /tmp/claude-ui.pid ] && ps -p $(cat /tmp/claude-ui.pid) > /dev/null 2>&1; then
        echo -e "  Claude Code UI: ${GREEN}● Running${NC} at http://localhost:3001"
    else
        echo -e "  Claude Code UI: ${RED}○ Stopped${NC}"
    fi
    
    # Check Claude CLI
    if command -v claude &> /dev/null; then
        echo -e "  Claude CLI: ${GREEN}✓ Installed${NC} at $(which claude)"
    else
        echo -e "  Claude CLI: ${RED}✗ Not found${NC}"
    fi
    ;;
    
  6)
    echo -e "\n${BLUE}Mobile Access URLs:${NC}"
    echo ""
    echo -e "  ${GREEN}Local Network:${NC} http://${LOCAL_IP}:3001"
    echo -e "  ${GREEN}Localhost:${NC} http://localhost:3001"
    echo ""
    echo -e "  ${YELLOW}QR Code for mobile:${NC}"
    echo "  ┌─────────────────────────┐"
    echo "  │ Scan with phone camera  │"
    echo "  │ to open Claude Code UI  │"
    echo "  └─────────────────────────┘"
    echo ""
    echo -e "  URL: http://${LOCAL_IP}:3001"
    
    # Optional: Generate actual QR code if qrencode is installed
    if command -v qrencode &> /dev/null; then
        qrencode -t ANSIUTF8 "http://${LOCAL_IP}:3001"
    fi
    ;;
    
  *)
    echo -e "${RED}Invalid choice${NC}"
    exit 1
    ;;
esac

echo ""
echo -e "${BLUE}─────────────────────────────${NC}"
echo -e "Press ${YELLOW}Ctrl+C${NC} to exit"