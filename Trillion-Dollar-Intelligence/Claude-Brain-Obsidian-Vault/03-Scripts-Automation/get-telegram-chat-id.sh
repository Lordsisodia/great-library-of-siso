#!/bin/bash

# Quick script to get your Telegram chat ID

BOT_TOKEN="8397799016:AAHdfdLx9Qqa8j8uhuYy9qEmFlkeH59dr_w"

echo "🤖 Getting your Telegram Chat ID..."
echo ""
echo "1. First, open Telegram and send a message to your bot:"
echo "   👉 https://t.me/Sisoprompt_bot"
echo ""
echo "2. Send any message like 'Hello' to the bot"
echo ""
echo "3. Press Enter here after sending the message..."
read -r

echo ""
echo "Fetching updates..."
echo ""

# Get updates
RESPONSE=$(curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getUpdates")

# Extract chat IDs
CHAT_IDS=$(echo "$RESPONSE" | grep -oE '"chat":\{"id":[0-9-]+' | grep -oE '[0-9-]+' | sort -u)

if [[ -n "$CHAT_IDS" ]]; then
    echo "✅ Found Chat ID(s):"
    echo "$CHAT_IDS" | while read -r id; do
        echo "   📱 $id"
    done
    echo ""
    echo "Your chat ID is likely: $(echo "$CHAT_IDS" | head -1)"
    echo ""
    echo "To update the notifier with this ID, run:"
    echo "sed -i '' 's/YOUR_CHAT_ID_HERE/$(echo "$CHAT_IDS" | head -1)/' ~/.claude/scripts/telegram-optimizer-notifier.sh"
else
    echo "❌ No chat ID found. Make sure you:"
    echo "   1. Started a chat with the bot"
    echo "   2. Sent at least one message"
    echo ""
    echo "Raw response:"
    echo "$RESPONSE" | head -100
fi