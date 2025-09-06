#!/bin/bash
# YouTube Channel Video Fetcher - Complete Usage Examples
# This script demonstrates how to fetch and process YouTube videos for AI analysis

set -e  # Exit on any error

echo "🎬 YouTube Channel Video Fetcher - Example Usage"
echo "==============================================="

# Check if API key is set
if [ -z "$YOUTUBE_API_KEY" ]; then
    echo "❌ Error: YOUTUBE_API_KEY environment variable not set"
    echo "📝 Please set it with: export YOUTUBE_API_KEY='your_api_key_here'"
    echo "🔑 Get API key from: https://console.cloud.google.com/"
    exit 1
fi

echo "✅ YouTube API key found"

# Create test directory
TEST_DIR="./youtube-test-output"
mkdir -p $TEST_DIR

echo "📁 Created test directory: $TEST_DIR"

# Example 1: Fetch coding/AI channel (use a known good channel)
echo ""
echo "🚀 Example 1: Fetching from a popular tech channel"
echo "Channel: Fireship (known for short, high-quality coding videos)"

node youtube-video-fetcher.js UC_ML5xP4uEBp6HMFPhsB7nA \
    --output=$TEST_DIR \
    --min-duration=60 \
    --max-duration=900 \
    --min-views=10000 \
    --sort=viewCount \
    || echo "⚠️  Channel fetch failed - continuing with examples"

# Example 2: Fetch specific video types
echo ""
echo "🚀 Example 2: Fetching tutorial videos only"
echo "Channel: 3Blue1Brown (excellent math/programming explanations)"

node youtube-video-fetcher.js UCYO_jab_esuFRV4b17AJtAw \
    --output=$TEST_DIR \
    --min-duration=300 \
    --min-views=50000 \
    --sort=duration \
    --published-after=2023-01-01 \
    || echo "⚠️  Channel fetch failed - continuing with examples"

# Example 3: Process for App Building Framework
echo ""
echo "🚀 Example 3: Fetch for App Building Framework analysis"
echo "Getting videos suitable for insight extraction"

# Create App Building Framework directory structure
FRAMEWORK_DIR="./THE-GREAT-LIBRARY-OF-SISO/App-Building-Framework/videos/youtube"
mkdir -p $FRAMEWORK_DIR

node youtube-video-fetcher.js UC_ML5xP4uEBp6HMFPhsB7nA \
    --output=$FRAMEWORK_DIR \
    --min-duration=180 \
    --min-views=5000 \
    --no-shorts \
    --sort=viewCount \
    || echo "⚠️  Framework channel fetch failed"

# Example 4: Extract transcripts from fetched videos
echo ""
echo "🚀 Example 4: Extract transcripts from fetched videos"

# Find the channel directory that was created
CHANNEL_DIR=$(find $TEST_DIR -name "*" -type d -maxdepth 1 | head -1)

if [ -d "$CHANNEL_DIR" ] && [ -f "$CHANNEL_DIR/videos-list.json" ]; then
    echo "📋 Found channel directory: $CHANNEL_DIR"
    
    # Install transcript extractor dependencies
    echo "📦 Installing transcript extractor dependencies..."
    pip3 install youtube-transcript-api || echo "⚠️  Transcript API installation failed"
    
    # Extract transcripts from first few videos
    echo "📝 Extracting transcripts..."
    python3 transcript-extractor.py --channel-dir "$CHANNEL_DIR" || echo "⚠️  Transcript extraction failed"
    
else
    echo "⚠️  No channel directory found - skipping transcript extraction"
fi

# Example 5: Show results and next steps
echo ""
echo "🎉 Examples Complete!"
echo "==================="

# Show directory structure
if [ -d "$TEST_DIR" ]; then
    echo "📁 Directory structure created:"
    find $TEST_DIR -type f -name "*.json" | head -10 | while read file; do
        echo "   📄 $file"
    done
    
    # Show stats from videos-list.json if available
    VIDEOS_FILE=$(find $TEST_DIR -name "videos-list.json" | head -1)
    if [ -f "$VIDEOS_FILE" ]; then
        echo ""
        echo "📊 Video Statistics:"
        python3 -c "
import json
with open('$VIDEOS_FILE', 'r') as f:
    videos = json.load(f)
    print(f'   📹 Total Videos: {len(videos)}')
    total_duration = sum(v['duration'] for v in videos)
    print(f'   ⏱️  Total Duration: {total_duration//3600}h {(total_duration%3600)//60}m')
    total_views = sum(v['viewCount'] for v in videos)
    print(f'   👀 Total Views: {total_views:,}')
    avg_duration = total_duration / len(videos) if videos else 0
    print(f'   📊 Average Duration: {avg_duration//60:.0f}m {avg_duration%60:.0f}s')
"
    fi
fi

echo ""
echo "🎯 Next Steps:"
echo "1. ✅ Videos fetched and organized"
echo "2. 📝 Transcripts extracted (if successful)"
echo "3. 🔍 Ready for AI analysis and insight extraction"
echo "4. 📚 Can be integrated into App Building Framework"

echo ""
echo "🛠️  Manual commands you can run:"
echo "   # Fetch specific channel"
echo "   node youtube-video-fetcher.js YOUR_CHANNEL_ID --min-duration=300"
echo ""
echo "   # Extract transcripts"  
echo "   python3 transcript-extractor.py --batch videos-list.json"
echo ""
echo "   # Analyze with AI (coming soon)"
echo "   node video-insights-analyzer.js videos-list.json"

echo ""
echo "✨ YouTube Video Fetcher setup complete!"