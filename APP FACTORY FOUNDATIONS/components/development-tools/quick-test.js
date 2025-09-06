#!/usr/bin/env node
/**
 * Quick Test Script for YouTube Video Fetcher
 * Tests the fetcher with a small, known-good channel
 */

const YouTubeVideoFetcher = require('./youtube-video-fetcher.js');

async function quickTest() {
  console.log('🧪 YouTube Video Fetcher - Quick Test');
  console.log('====================================');
  
  // Check for API key
  const apiKey = process.env.YOUTUBE_API_KEY;
  if (!apiKey) {
    console.error('❌ Error: YOUTUBE_API_KEY environment variable not set');
    console.log('🔑 Get API key from: https://console.cloud.google.com/');
    console.log('💡 Then run: export YOUTUBE_API_KEY="your_key_here"');
    process.exit(1);
  }
  
  console.log('✅ API key found');
  
  try {
    const fetcher = new YouTubeVideoFetcher(apiKey);
    
    // Test with Fireship channel (known for short, high-quality videos)
    const channelId = 'UC_ML5xP4uEBp6HMFPhsB7nA';
    console.log(`🔍 Testing with channel: ${channelId}`);
    
    // Fetch just a few recent videos for testing
    const result = await fetcher.getAllVideosFromChannel(channelId, {
      minDuration: 60,       // At least 1 minute
      maxDuration: 600,      // At most 10 minutes  
      minViews: 10000,       // At least 10k views
      sortBy: 'publishedAt', // Most recent first
      saveMetadata: false    // Don't save files for test
    });
    
    console.log('\n🎉 Test Results:');
    console.log(`📺 Channel: ${result.channelInfo.title}`);
    console.log(`📊 Videos Found: ${result.videos.length}`);
    console.log(`⏱️  Total Duration: ${Math.round(result.summary.totalDuration / 60)} minutes`);
    console.log(`👀 Total Views: ${result.summary.totalViews.toLocaleString()}`);
    
    if (result.videos.length > 0) {
      console.log('\n📋 Sample Videos:');
      result.videos.slice(0, 5).forEach((video, i) => {
        console.log(`   ${i + 1}. ${video.title}`);
        console.log(`      📅 ${video.publishedAt.split('T')[0]} | ⏱️ ${video.durationFormatted} | 👀 ${video.viewCount.toLocaleString()}`);
      });
      
      console.log('\n✅ Test PASSED - Video fetcher working correctly!');
      console.log('\n🚀 Ready to use with real channels:');
      console.log('   node youtube-video-fetcher.js CHANNEL_ID --min-duration=300');
    } else {
      console.log('⚠️  No videos found with current filters');
      console.log('💡 Try reducing minViews or minDuration');
    }
    
  } catch (error) {
    console.error('❌ Test FAILED:', error.message);
    
    if (error.message.includes('quotaExceeded')) {
      console.log('💡 Solution: YouTube API quota exceeded. Try again tomorrow.');
    } else if (error.message.includes('keyInvalid')) {
      console.log('💡 Solution: Invalid API key. Check your YOUTUBE_API_KEY');
    } else {
      console.log('💡 Solution: Check your internet connection and API key');
    }
    
    process.exit(1);
  }
}

// Run test if called directly
if (require.main === module) {
  quickTest().catch(console.error);
}

module.exports = quickTest;