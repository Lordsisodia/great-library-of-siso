#!/bin/bash

BOT_TOKEN="8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w"

echo "🤖 Live Chat ID Finder"
echo "====================="
echo ""
echo "1. Open Telegram: https://t.me/Sisoprompt_bot"
echo "2. Click 'START' or send any message"
echo "3. Your chat ID will appear here automatically!"
echo ""
echo "Waiting for messages..."
echo ""

while true; do
    RESPONSE=$(curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getUpdates")
    
    # Check if we have results
    if [[ "$RESPONSE" =~ \"chat\":\{\"id\":([0-9-]+) ]]; then
        CHAT_ID="${BASH_REMATCH[1]}"
        echo ""
        echo "✅ FOUND YOUR CHAT ID: $CHAT_ID"
        echo ""
        echo "Updating notifier automatically..."
        
        # Update the notifier
        sed -i '' "s/YOUR_CHAT_ID_HERE/$CHAT_ID/" ~/.claude/scripts/telegram-optimizer-notifier-v2.sh
        
        echo "✅ Done! Your notifier is configured!"
        echo ""
        echo "To start receiving notifications, run:"
        echo "~/.claude/scripts/telegram-optimizer-notifier-v2.sh &"
        echo ""
        
        # Send test message
        curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
            -d "chat_id=${CHAT_ID}" \
            -d "text=✅ Bot configured successfully! You'll now receive prompt optimization notifications here." \
            > /dev/null
            
        break
    else
        echo -n "."
        sleep 2
    fi
done