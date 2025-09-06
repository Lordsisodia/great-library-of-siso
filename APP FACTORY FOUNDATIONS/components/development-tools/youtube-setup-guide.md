# YouTube Channel Video Fetcher - Setup Guide

Complete guide to fetch all videos from any YouTube channel for transcript extraction and AI analysis.

## 🚀 Quick Start

### 1. Get YouTube API Key

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create New Project** (or select existing)
   - Click "Select a project" → "New Project"
   - Name: "YouTube Video Fetcher" 
   - Click "Create"

3. **Enable YouTube Data API v3**
   - Go to "APIs & Services" → "Library"
   - Search for "YouTube Data API v3"
   - Click on it → "Enable"

4. **Create API Key**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - **Copy the API key** - you'll need this!

5. **Optional: Restrict API Key** (Recommended)
   - Click on your API key to edit
   - Under "API restrictions" select "Restrict key"
   - Choose "YouTube Data API v3"
   - Save

### 2. Install Dependencies

```bash
# Make script executable
chmod +x youtube-video-fetcher.js

# Install Node.js dependencies (if using as module)
npm install
```

### 3. Set Environment Variable

```bash
# Option 1: Export in terminal
export YOUTUBE_API_KEY="your_api_key_here"

# Option 2: Add to .env file
echo "YOUTUBE_API_KEY=your_api_key_here" >> .env

# Option 3: Add to .bashrc/.zshrc for permanent
echo 'export YOUTUBE_API_KEY="your_api_key_here"' >> ~/.bashrc
```

## 📋 Usage Examples

### Basic Channel Fetch
```bash
# Using channel ID (recommended)
node youtube-video-fetcher.js UCBJycsmduvYEL83R_U4JriQ

# Using channel handle 
node youtube-video-fetcher.js @MarquesB

# With API key parameter
node youtube-video-fetcher.js UCxxxxxx --api-key=your_key_here
```

### Advanced Filtering
```bash
# Only videos longer than 5 minutes with 10k+ views
node youtube-video-fetcher.js UCxxxxxx --min-duration=300 --min-views=10000

# Exclude YouTube Shorts, sort by view count
node youtube-video-fetcher.js UCxxxxxx --no-shorts --sort=viewCount

# Videos from 2024 only, save to specific folder
node youtube-video-fetcher.js UCxxxxxx \
  --published-after=2024-01-01 \
  --published-before=2024-12-31 \
  --output=./2024-videos/

# Long-form content analysis ready videos
node youtube-video-fetcher.js UCxxxxxx \
  --min-duration=600 \
  --max-duration=3600 \
  --min-views=5000 \
  --no-shorts \
  --sort=duration
```

### For App Building Framework Analysis
```bash
# Get AI/coding videos suitable for insight extraction
node youtube-video-fetcher.js UCxxxxxx \
  --min-duration=300 \
  --min-views=1000 \
  --output=./THE-GREAT-LIBRARY-OF-SISO/App-Building-Framework/videos/youtube/ \
  --sort=viewCount
```

## 🔍 Finding Channel IDs

### Method 1: Channel URL
**Format**: `https://www.youtube.com/channel/UCxxxxxx`
- The part after `/channel/` is the channel ID

### Method 2: Channel Handle
**Format**: `https://www.youtube.com/@channelname`
- Use `@channelname` directly in the script

### Method 3: Any YouTube Video
1. Go to any video from the channel
2. Click on the channel name
3. Look at URL - either shows channel ID or handle

### Method 4: Browser Developer Tools
1. Go to channel page
2. View page source (Ctrl+U)
3. Search for `"channelId":"` 
4. Copy the ID after the colon

## 📊 Output Structure

The script creates this file structure:

```
videos/
└── channel-name/
    ├── channel-info.json          # Channel metadata
    ├── videos-list.json           # All video data
    ├── process-videos.sh          # Auto-generated processing script
    └── videos/                    # Individual video files
        ├── video-title-1.json     # Ready for transcript extraction
        ├── video-title-2.json
        └── ...
```

### Sample Output Files

**channel-info.json**:
```json
{
  "id": "UCxxxxxx",
  "title": "Channel Name",
  "description": "Channel description...",
  "customUrl": "@channelname",
  "publishedAt": "2010-01-01T00:00:00Z"
}
```

**videos-list.json**:
```json
[
  {
    "id": "dQw4w9WgXcQ",
    "title": "Video Title",
    "description": "Video description...",
    "publishedAt": "2024-01-01T00:00:00Z",
    "duration": 300,
    "durationFormatted": "5:00",
    "viewCount": 1000000,
    "likeCount": 50000,
    "tags": ["ai", "coding", "tutorial"],
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "isShort": false,
    "processed": false
  }
]
```

## 🤖 Integration with App Building Framework

### Step 1: Fetch Videos
```bash
# Target AI/coding channels for insights
node youtube-video-fetcher.js UCxxxxxx \
  --min-duration=600 \
  --output=./THE-GREAT-LIBRARY-OF-SISO/App-Building-Framework/videos/youtube/
```

### Step 2: Extract Transcripts
```bash
# Use the auto-generated processing script
cd ./THE-GREAT-LIBRARY-OF-SISO/App-Building-Framework/videos/youtube/channel-name/
./process-videos.sh
```

### Step 3: Analyze with AI
```bash
# Process each video for insights (create this next)
node ../../../components/development-tools/extract-video-insights.js videos-list.json
```

## 📈 API Quota Management

### Quota Costs (per request)
- **playlistItems.list**: 2 quota points ⭐ (efficient)
- **videos.list**: 1 quota point ⭐ (efficient)  
- **search.list**: 100 quota points ❌ (avoid)
- **channels.list**: 1 quota point ⭐

### Daily Quota Limits
- **Default**: 10,000 quota points per day
- **Typical channel fetch**: 50-200 quota points (very efficient!)

### Optimization Tips
1. **Use playlist approach** (this script) vs search API
2. **Batch video details** (50 videos per request)
3. **Cache results** - save output to avoid re-fetching
4. **Add delays** between requests (script includes this)

### Calculate Quota Usage
```javascript
// For a channel with 1000 videos:
const quotaUsed = 
  1 +                    // Get channel info
  Math.ceil(1000/50) +   // Fetch playlist items (20 requests)
  Math.ceil(1000/50);    // Get video details (20 requests)
// Total: ~41 quota points for 1000 videos!
```

## 🛠️ Customization Options

### Add New Filters
Edit the script to add custom filtering:

```javascript
// In getAllVideosFromChannel method
const processedVideos = videoDetails.map(video => {
  // Add custom fields
  return {
    ...video,
    // Custom analysis
    hasGoodThumbnail: video.thumbnails.high ? true : false,
    likeRatio: video.statistics.likeCount / video.statistics.viewCount,
    titleLength: video.snippet.title.length,
    // AI readiness score
    aiReadinessScore: calculateAIReadiness(video)
  };
}).filter(video => {
  // Add custom filters
  if (video.titleLength < 10) return false; // Too short title
  if (video.likeRatio < 0.01) return false; // Poor engagement
  return true;
});
```

### Add Transcript Integration
```javascript
// In processedVideos.map()
transcriptUrl: `https://www.youtube.com/watch?v=${video.id}`, 
transcriptAvailable: await checkTranscriptAvailable(video.id),
transcriptLanguage: await getTranscriptLanguage(video.id)
```

## 🔧 Troubleshooting

### Common Issues

**❌ "API key required"**
```bash
# Solution: Set environment variable
export YOUTUBE_API_KEY="your_key_here"
# Or use --api-key parameter
```

**❌ "YouTube API Error: quotaExceeded"**
```bash
# Solution: Wait until next day or request quota increase
# Check usage: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas
```

**❌ "Channel not found"**
```bash
# Solution: Verify channel ID/handle
# Try: https://www.youtube.com/channel/YOUR_CHANNEL_ID
# Or: https://www.youtube.com/@channelname
```

**❌ "Permission denied" on script**
```bash
chmod +x youtube-video-fetcher.js
```

### Debug Mode
Add debug output to script:
```bash
DEBUG=true node youtube-video-fetcher.js UCxxxxxx
```

### Test with Small Channel First
```bash
# Test with a channel that has <50 videos
node youtube-video-fetcher.js UCsmallchannel --output=./test-output/
```

## 🚀 Next Steps

1. **Set up transcript extraction** with youtube-transcript-api
2. **Create insight analysis** using AI APIs  
3. **Integrate with App Building Framework** workflow
4. **Build batch processing** for multiple channels
5. **Add real-time monitoring** for new videos

This fetcher is the first step in creating a comprehensive video analysis pipeline for the App Building Framework!