#!/bin/bash

echo "🔍 TELEGRAM CHAT ID FINDER"
echo "========================="
echo ""
echo "Checking for any messages..."

# Try with offset to get all messages
for offset in -1 0 1; do
    echo "Checking offset $offset..."
    RESPONSE=$(curl -s "https://api.telegram.org/bot8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w/getUpdates?offset=$offset")
    
    if [[ "$RESPONSE" =~ \"chat\":\{\"id\":([0-9-]+) ]]; then
        CHAT_ID="${BASH_REMATCH[1]}"
        echo ""
        echo "✅ FOUND YOUR CHAT ID: $CHAT_ID"
        echo ""
        
        # Update the notifier
        sed -i '' "s/YOUR_CHAT_ID_HERE/$CHAT_ID/" ~/.claude/scripts/telegram-optimizer-notifier-v2.sh
        
        echo "✅ Notifier updated!"
        echo ""
        echo "Sending test message..."
        
        curl -s -X POST "https://api.telegram.org/bot8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w/sendMessage" \
            -d "chat_id=${CHAT_ID}" \
            -d "text=🎉 Success! Your Claude Optimizer notifications are now configured!" \
            -d "parse_mode=Markdown"
            
        echo ""
        echo "To start notifications: ~/.claude/scripts/telegram-optimizer-notifier-v2.sh &"
        exit 0
    fi
done

echo ""
echo "❌ No messages found yet!"
echo ""
echo "IMPORTANT: You must:"
echo "1. Go to: https://t.me/Sisoprompt_bot"
echo "2. Click the 'START' button (blue button at bottom)"
echo "3. Send any message"
echo ""
echo "Alternative: Find your ID manually:"
echo "1. Search for @userinfobot on Telegram"
echo "2. Start a chat with it"
echo "3. It will show your user ID"
echo "4. Update manually: sed -i '' 's/YOUR_CHAT_ID_HERE/YOUR_ID/' ~/.claude/scripts/telegram-optimizer-notifier-v2.sh"