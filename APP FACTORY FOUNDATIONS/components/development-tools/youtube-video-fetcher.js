#!/usr/bin/env node
/**
 * YouTube Channel Video Fetcher
 * Fetches all videos from a YouTube channel and saves them for processing
 * 
 * Usage: 
 * node youtube-video-fetcher.js <channel-id> [--save-transcripts] [--min-duration=300]
 * 
 * Examples:
 * node youtube-video-fetcher.js UCxxxxxx --save-transcripts --min-duration=600
 * node youtube-video-fetcher.js @channelname --output=./videos/
 */

const fs = require('fs');
const path = require('path');

class YouTubeVideoFetcher {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.baseUrl = 'https://www.googleapis.com/youtube/v3';
    this.videoList = [];
  }

  /**
   * Get channel's uploads playlist ID
   */
  async getChannelUploadsPlaylistId(channelId) {
    // Handle @channel format
    if (channelId.startsWith('@')) {
      channelId = await this.getChannelIdFromHandle(channelId);
    }

    const url = `${this.baseUrl}/channels?id=${channelId}&key=${this.apiKey}&part=contentDetails,snippet`;
    
    try {
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.error) {
        throw new Error(`YouTube API Error: ${data.error.message}`);
      }
      
      if (data.items && data.items.length > 0) {
        return {
          uploadsPlaylistId: data.items[0].contentDetails.relatedPlaylists.uploads,
          channelInfo: {
            id: data.items[0].id,
            title: data.items[0].snippet.title,
            description: data.items[0].snippet.description,
            customUrl: data.items[0].snippet.customUrl || null,
            publishedAt: data.items[0].snippet.publishedAt
          }
        };
      }
      throw new Error('Channel not found');
    } catch (error) {
      console.error('❌ Error fetching channel details:', error.message);
      throw error;
    }
  }

  /**
   * Convert @channelname to channel ID
   */
  async getChannelIdFromHandle(handle) {
    // First try to search for the channel
    const searchUrl = `${this.baseUrl}/search?q=${handle.substring(1)}&type=channel&key=${this.apiKey}&part=id`;
    
    try {
      const response = await fetch(searchUrl);
      const data = await response.json();
      
      if (data.items && data.items.length > 0) {
        return data.items[0].id.channelId;
      }
      throw new Error(`Channel handle ${handle} not found`);
    } catch (error) {
      console.error('❌ Error converting handle to channel ID:', error.message);
      throw error;
    }
  }

  /**
   * Fetch all videos from uploads playlist with pagination
   */
  async fetchAllVideosFromPlaylist(playlistId, options = {}) {
    const {
      minDuration = 0, // seconds
      maxResults = 50,
      publishedAfter = null,
      publishedBefore = null
    } = options;

    const videos = [];
    let nextPageToken = '';
    let totalFetched = 0;

    console.log(`🔍 Fetching videos from playlist: ${playlistId}`);
    
    do {
      const url = `${this.baseUrl}/playlistItems?` +
        `playlistId=${playlistId}&` +
        `key=${this.apiKey}&` +
        `part=snippet&` +
        `maxResults=${maxResults}` +
        `${nextPageToken ? `&pageToken=${nextPageToken}` : ''}`;
      
      try {
        console.log(`📥 Fetching page ${Math.floor(totalFetched / maxResults) + 1}...`);
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.error) {
          throw new Error(`YouTube API Error: ${data.error.message}`);
        }
        
        if (data.items) {
          // Filter videos by date if specified
          let pageVideos = data.items.filter(item => {
            const publishedAt = new Date(item.snippet.publishedAt);
            
            if (publishedAfter && publishedAt < new Date(publishedAfter)) return false;
            if (publishedBefore && publishedAt > new Date(publishedBefore)) return false;
            
            return true;
          });
          
          videos.push(...pageVideos);
          totalFetched += pageVideos.length;
          
          console.log(`📊 Progress: ${totalFetched} videos fetched`);
        }
        
        nextPageToken = data.nextPageToken || null;
        
        // Add small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 100));
        
      } catch (error) {
        console.error('❌ Error fetching playlist items:', error.message);
        break;
      }
    } while (nextPageToken);
    
    console.log(`✅ Total videos fetched: ${videos.length}`);
    return videos;
  }

  /**
   * Get detailed video information including duration, views, etc.
   */
  async getVideoDetails(videoIds) {
    if (!videoIds.length) return [];

    // YouTube API allows max 50 video IDs per request
    const chunks = [];
    for (let i = 0; i < videoIds.length; i += 50) {
      chunks.push(videoIds.slice(i, i + 50));
    }
    
    const allDetails = [];
    
    console.log(`📋 Getting detailed info for ${videoIds.length} videos...`);
    
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const url = `${this.baseUrl}/videos?` +
        `part=contentDetails,snippet,statistics&` +
        `id=${chunk.join(',')}&` +
        `key=${this.apiKey}`;
      
      try {
        console.log(`📥 Fetching details batch ${i + 1}/${chunks.length}`);
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.error) {
          throw new Error(`YouTube API Error: ${data.error.message}`);
        }
        
        if (data.items) {
          allDetails.push(...data.items);
        }
        
        // Add delay between requests
        await new Promise(resolve => setTimeout(resolve, 200));
        
      } catch (error) {
        console.error(`❌ Error fetching video details for batch ${i + 1}:`, error.message);
      }
    }
    
    return allDetails;
  }

  /**
   * Parse ISO 8601 duration to seconds
   */
  parseDuration(duration) {
    const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
    if (!match) return 0;
    
    const hours = parseInt(match[1] || '0');
    const minutes = parseInt(match[2] || '0');
    const seconds = parseInt(match[3] || '0');
    
    return hours * 3600 + minutes * 60 + seconds;
  }

  /**
   * Main method to get all videos from channel with filtering
   */
  async getAllVideosFromChannel(channelId, options = {}) {
    const {
      minDuration = 0,
      maxDuration = Infinity,
      minViews = 0,
      includeShorts = true,
      sortBy = 'publishedAt', // publishedAt, viewCount, duration
      outputDir = './videos/',
      saveMetadata = true
    } = options;

    try {
      console.log(`🚀 Starting YouTube video fetch for channel: ${channelId}`);
      
      // Get channel info and uploads playlist
      const { uploadsPlaylistId, channelInfo } = await this.getChannelUploadsPlaylistId(channelId);
      console.log(`📺 Channel: ${channelInfo.title}`);
      console.log(`📁 Upload Playlist ID: ${uploadsPlaylistId}`);
      
      // Fetch all videos from playlist
      const playlistVideos = await this.fetchAllVideosFromPlaylist(uploadsPlaylistId, options);
      
      if (playlistVideos.length === 0) {
        console.log('ℹ️ No videos found in channel');
        return { channelInfo, videos: [] };
      }

      // Extract video IDs
      const videoIds = playlistVideos.map(item => item.snippet.resourceId.videoId);
      
      // Get detailed video information
      const videoDetails = await this.getVideoDetails(videoIds);
      
      // Process and filter videos
      const processedVideos = videoDetails.map(video => {
        const duration = this.parseDuration(video.contentDetails.duration);
        const viewCount = parseInt(video.statistics.viewCount || '0');
        const likeCount = parseInt(video.statistics.likeCount || '0');
        
        return {
          id: video.id,
          title: video.snippet.title,
          description: video.snippet.description,
          publishedAt: video.snippet.publishedAt,
          duration: duration,
          durationFormatted: this.formatDuration(duration),
          viewCount: viewCount,
          likeCount: likeCount,
          tags: video.snippet.tags || [],
          categoryId: video.snippet.categoryId,
          thumbnails: video.snippet.thumbnails,
          url: `https://www.youtube.com/watch?v=${video.id}`,
          isShort: duration < 60, // Consider videos under 60s as shorts
          // Metadata for processing
          transcriptUrl: null, // Will be filled later
          processed: false,
          insights: null
        };
      }).filter(video => {
        // Apply filters
        if (video.duration < minDuration) return false;
        if (video.duration > maxDuration) return false;
        if (video.viewCount < minViews) return false;
        if (!includeShorts && video.isShort) return false;
        
        return true;
      });

      // Sort videos
      processedVideos.sort((a, b) => {
        switch (sortBy) {
          case 'viewCount':
            return b.viewCount - a.viewCount;
          case 'duration':
            return b.duration - a.duration;
          case 'publishedAt':
          default:
            return new Date(b.publishedAt) - new Date(a.publishedAt);
        }
      });

      console.log(`✅ Processed ${processedVideos.length} videos (filtered from ${videoDetails.length})`);
      
      // Save results if requested
      if (saveMetadata) {
        await this.saveResults(channelInfo, processedVideos, outputDir);
      }

      return {
        channelInfo,
        videos: processedVideos,
        summary: {
          totalVideos: processedVideos.length,
          totalDuration: processedVideos.reduce((sum, v) => sum + v.duration, 0),
          averageDuration: processedVideos.length > 0 ? 
            processedVideos.reduce((sum, v) => sum + v.duration, 0) / processedVideos.length : 0,
          totalViews: processedVideos.reduce((sum, v) => sum + v.viewCount, 0),
          dateRange: {
            earliest: processedVideos.length > 0 ? 
              new Date(Math.min(...processedVideos.map(v => new Date(v.publishedAt)))).toISOString() : null,
            latest: processedVideos.length > 0 ?
              new Date(Math.max(...processedVideos.map(v => new Date(v.publishedAt)))).toISOString() : null
          }
        }
      };

    } catch (error) {
      console.error('❌ Error in getAllVideosFromChannel:', error.message);
      throw error;
    }
  }

  /**
   * Format duration in seconds to readable format
   */
  formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Save results to files
   */
  async saveResults(channelInfo, videos, outputDir) {
    // Create output directory
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Create channel-specific directory
    const channelDir = path.join(outputDir, this.sanitizeFilename(channelInfo.title));
    if (!fs.existsSync(channelDir)) {
      fs.mkdirSync(channelDir, { recursive: true });
    }

    // Save channel metadata
    const channelFile = path.join(channelDir, 'channel-info.json');
    fs.writeFileSync(channelFile, JSON.stringify(channelInfo, null, 2));
    console.log(`💾 Saved channel info: ${channelFile}`);

    // Save videos list
    const videosFile = path.join(channelDir, 'videos-list.json');
    fs.writeFileSync(videosFile, JSON.stringify(videos, null, 2));
    console.log(`💾 Saved videos list: ${videosFile}`);

    // Create individual video files for future processing
    const videosDir = path.join(channelDir, 'videos');
    if (!fs.existsSync(videosDir)) {
      fs.mkdirSync(videosDir);
    }

    // Save processing template for each video
    for (const video of videos.slice(0, 10)) { // Limit to first 10 for demo
      const videoFile = path.join(videosDir, `${this.sanitizeFilename(video.title)}.json`);
      const videoTemplate = {
        ...video,
        processing: {
          transcriptRequested: false,
          transcriptCompleted: false,
          insightsExtracted: false,
          addedToFramework: false
        },
        notes: "Video ready for transcript extraction and insight analysis"
      };
      
      fs.writeFileSync(videoFile, JSON.stringify(videoTemplate, null, 2));
    }

    console.log(`💾 Created video templates in: ${videosDir}`);

    // Generate processing script
    const scriptContent = this.generateProcessingScript(channelInfo, videos);
    const scriptFile = path.join(channelDir, 'process-videos.sh');
    fs.writeFileSync(scriptFile, scriptContent);
    fs.chmodSync(scriptFile, '755');
    console.log(`📝 Generated processing script: ${scriptFile}`);
  }

  /**
   * Generate processing script for batch video analysis
   */
  generateProcessingScript(channelInfo, videos) {
    return `#!/bin/bash
# Auto-generated video processing script for ${channelInfo.title}
# Generated on ${new Date().toISOString()}

echo "🚀 Processing ${videos.length} videos from ${channelInfo.title}"

# Process each video for transcript extraction
for i in {1..${Math.min(videos.length, 20)}}; do
  VIDEO_ID=\${videos[\$i-1].id}
  VIDEO_TITLE=\${videos[\$i-1].title}
  
  echo "📹 Processing: \$VIDEO_TITLE"
  
  # Extract transcript using youtube-transcript-api or similar
  # python3 transcript-extractor.py \$VIDEO_ID
  
  # Extract insights using AI
  # node extract-insights.js "./videos/\$VIDEO_TITLE.json"
  
  echo "✅ Completed: \$VIDEO_TITLE"
done

echo "🎉 All videos processed!"
echo "📊 Total processed: ${videos.length} videos"
echo "⏱️  Total duration: ${videos.reduce((sum, v) => sum + v.duration, 0)} seconds"
`;
  }

  /**
   * Sanitize filename for saving
   */
  sanitizeFilename(filename) {
    return filename
      .replace(/[^\w\s-]/g, '') // Remove special chars except word chars, spaces, hyphens
      .replace(/\s+/g, '-') // Replace spaces with hyphens
      .toLowerCase()
      .substring(0, 100); // Limit length
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log(`
🎬 YouTube Channel Video Fetcher

Usage: node youtube-video-fetcher.js <channel-id> [options]

Options:
  --api-key=KEY           YouTube Data API key (or set YOUTUBE_API_KEY env var)
  --output=DIR           Output directory (default: ./videos/)
  --min-duration=SECS    Minimum video duration in seconds
  --max-duration=SECS    Maximum video duration in seconds  
  --min-views=NUM        Minimum view count
  --no-shorts           Exclude YouTube Shorts (< 60s)
  --sort=FIELD          Sort by: publishedAt, viewCount, duration
  --published-after=DATE Only videos after this date (YYYY-MM-DD)
  --published-before=DATE Only videos before this date (YYYY-MM-DD)

Examples:
  node youtube-video-fetcher.js UCxxxxxx --min-duration=300 --output=./ai-videos/
  node youtube-video-fetcher.js @channelname --no-shorts --min-views=1000
    `);
    process.exit(1);
  }

  const channelId = args[0];
  const options = {};
  
  // Parse command line arguments
  args.slice(1).forEach(arg => {
    if (arg.startsWith('--')) {
      const [key, value] = arg.substring(2).split('=');
      switch (key) {
        case 'api-key':
          options.apiKey = value;
          break;
        case 'output':
          options.outputDir = value;
          break;
        case 'min-duration':
          options.minDuration = parseInt(value);
          break;
        case 'max-duration':
          options.maxDuration = parseInt(value);
          break;
        case 'min-views':
          options.minViews = parseInt(value);
          break;
        case 'no-shorts':
          options.includeShorts = false;
          break;
        case 'sort':
          options.sortBy = value;
          break;
        case 'published-after':
          options.publishedAfter = value;
          break;
        case 'published-before':
          options.publishedBefore = value;
          break;
      }
    }
  });

  // Get API key from argument or environment
  const apiKey = options.apiKey || process.env.YOUTUBE_API_KEY;
  
  if (!apiKey) {
    console.error('❌ YouTube API key required. Set YOUTUBE_API_KEY env var or use --api-key=KEY');
    process.exit(1);
  }

  try {
    const fetcher = new YouTubeVideoFetcher(apiKey);
    const result = await fetcher.getAllVideosFromChannel(channelId, options);
    
    console.log(`
🎉 Fetch Complete!

📺 Channel: ${result.channelInfo.title}
📊 Videos Found: ${result.summary.totalVideos}
⏱️  Total Duration: ${Math.round(result.summary.totalDuration / 3600)} hours
👀 Total Views: ${result.summary.totalViews.toLocaleString()}
📅 Date Range: ${result.summary.dateRange.earliest?.split('T')[0]} to ${result.summary.dateRange.latest?.split('T')[0]}

📁 Files saved to: ${options.outputDir || './videos/'}

Next Steps:
1. Review the videos-list.json file
2. Run the generated process-videos.sh script  
3. Extract transcripts and insights for App Building Framework
    `);

  } catch (error) {
    console.error('❌ Fatal error:', error.message);
    process.exit(1);
  }
}

// Export for use as module
if (require.main === module) {
  main();
} else {
  module.exports = YouTubeVideoFetcher;
}