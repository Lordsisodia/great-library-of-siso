#!/usr/bin/env python3
"""
YouTube Transcript Extractor
Extracts transcripts from YouTube videos and saves them for AI analysis

Requirements:
pip install youtube-transcript-api

Usage:
python3 transcript-extractor.py <video_id>
python3 transcript-extractor.py --batch videos-list.json
python3 transcript-extractor.py --channel-dir ./channel-name/
"""

import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter, JSONFormatter
except ImportError:
    print("❌ Error: youtube-transcript-api not installed")
    print("📦 Install with: pip install youtube-transcript-api")
    sys.exit(1)

class TranscriptExtractor:
    def __init__(self, output_dir="transcripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.text_formatter = TextFormatter()
        self.json_formatter = JSONFormatter()
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'no_transcript': 0
        }
    
    def extract_transcript(self, video_id, video_title=None, save_files=True):
        """
        Extract transcript for a single video
        Returns: dict with transcript data and metadata
        """
        try:
            print(f"📥 Extracting transcript for: {video_id}")
            
            # Try to get transcript in multiple languages (preference order)
            languages = ['en', 'en-US', 'en-GB', 'auto']  
            transcript_data = None
            language_used = None
            
            for lang in languages:
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    
                    # Try to find manually created transcript first
                    try:
                        transcript = transcript_list.find_manually_created_transcript([lang])
                        transcript_data = transcript.fetch()
                        language_used = lang
                        transcript_type = "manual"
                        break
                    except:
                        pass
                    
                    # Fall back to auto-generated
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        transcript_data = transcript.fetch()
                        language_used = lang
                        transcript_type = "auto-generated"
                        break
                    except:
                        continue
                        
                except Exception as e:
                    continue
            
            if not transcript_data:
                print(f"⚠️  No transcript available for {video_id}")
                self.stats['no_transcript'] += 1
                return {
                    'video_id': video_id,
                    'title': video_title,
                    'success': False,
                    'error': 'No transcript available',
                    'transcript': None
                }
            
            # Process transcript data
            transcript_text = self.text_formatter.format_transcript(transcript_data)
            
            # Calculate transcript statistics
            word_count = len(transcript_text.split())
            duration = transcript_data[-1]['start'] + transcript_data[-1]['duration'] if transcript_data else 0
            
            result = {
                'video_id': video_id,
                'title': video_title or f"Video {video_id}",
                'success': True,
                'language': language_used,
                'transcript_type': transcript_type,
                'extracted_at': datetime.now().isoformat(),
                'statistics': {
                    'word_count': word_count,
                    'duration_seconds': duration,
                    'segments_count': len(transcript_data),
                    'words_per_minute': (word_count / (duration / 60)) if duration > 0 else 0
                },
                'transcript_text': transcript_text,
                'transcript_segments': transcript_data
            }
            
            # Save files if requested
            if save_files:
                self.save_transcript_files(result)
            
            print(f"✅ Success: {word_count} words, {duration:.0f}s, {transcript_type}")
            self.stats['successful'] += 1
            return result
            
        except Exception as e:
            print(f"❌ Error extracting transcript for {video_id}: {str(e)}")
            self.stats['failed'] += 1
            return {
                'video_id': video_id,
                'title': video_title,
                'success': False,
                'error': str(e),
                'transcript': None
            }
        
        finally:
            self.stats['processed'] += 1
    
    def save_transcript_files(self, transcript_result):
        """Save transcript in multiple formats"""
        video_id = transcript_result['video_id']
        safe_title = self.sanitize_filename(transcript_result['title'])
        
        # Create video-specific directory
        video_dir = self.output_dir / f"{safe_title}_{video_id}"
        video_dir.mkdir(exist_ok=True)
        
        # Save full metadata as JSON
        metadata_file = video_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(transcript_result, f, indent=2, ensure_ascii=False)
        
        # Save clean text version
        text_file = video_dir / "transcript.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"# {transcript_result['title']}\n")
            f.write(f"# Video ID: {video_id}\n")
            f.write(f"# Extracted: {transcript_result['extracted_at']}\n")
            f.write(f"# Language: {transcript_result['language']}\n")
            f.write(f"# Type: {transcript_result['transcript_type']}\n")
            f.write(f"# Duration: {transcript_result['statistics']['duration_seconds']:.0f}s\n")
            f.write(f"# Words: {transcript_result['statistics']['word_count']}\n\n")
            f.write(transcript_result['transcript_text'])
        
        # Save timestamped version for AI analysis
        timestamped_file = video_dir / "transcript-timestamped.json"
        with open(timestamped_file, 'w', encoding='utf-8') as f:
            json.dump({
                'video_id': video_id,
                'title': transcript_result['title'],
                'segments': transcript_result['transcript_segments']
            }, f, indent=2)
        
        # Create AI analysis template
        analysis_template = {
            'video_info': {
                'id': video_id,
                'title': transcript_result['title'],
                'url': f"https://www.youtube.com/watch?v={video_id}"
            },
            'transcript_summary': {
                'word_count': transcript_result['statistics']['word_count'],
                'duration': transcript_result['statistics']['duration_seconds'],
                'estimated_reading_time': transcript_result['statistics']['word_count'] // 200
            },
            'ai_analysis': {
                'key_insights': [],
                'main_topics': [],
                'actionable_takeaways': [],
                'code_examples': [],
                'tools_mentioned': [],
                'frameworks_discussed': [],
                'best_practices': [],
                'workflow_patterns': []
            },
            'processing_notes': {
                'extracted_at': transcript_result['extracted_at'],
                'ready_for_ai_analysis': True,
                'analysis_completed': False
            }
        }
        
        analysis_file = video_dir / "ai-analysis-template.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_template, f, indent=2)
        
        print(f"💾 Saved transcript files to: {video_dir}")
    
    def extract_from_video_list(self, videos_list_file):
        """Extract transcripts from a videos-list.json file"""
        try:
            with open(videos_list_file, 'r', encoding='utf-8') as f:
                videos = json.load(f)
            
            if not isinstance(videos, list):
                print("❌ Error: videos-list.json should contain an array of video objects")
                return
            
            print(f"🚀 Processing {len(videos)} videos from {videos_list_file}")
            
            results = []
            for i, video in enumerate(videos, 1):
                print(f"\n📹 [{i}/{len(videos)}] Processing: {video.get('title', 'Unknown')}")
                
                result = self.extract_transcript(
                    video['id'], 
                    video.get('title'),
                    save_files=True
                )
                results.append(result)
                
                # Add small delay to be respectful
                import time
                time.sleep(0.5)
            
            # Save batch results
            batch_results_file = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(batch_results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_at': datetime.now().isoformat(),
                    'source_file': str(videos_list_file),
                    'statistics': self.stats,
                    'results': results
                }, f, indent=2)
            
            print(f"\n📊 Batch processing complete!")
            print(f"💾 Results saved to: {batch_results_file}")
            self.print_stats()
            
        except FileNotFoundError:
            print(f"❌ Error: File {videos_list_file} not found")
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in {videos_list_file}")
        except Exception as e:
            print(f"❌ Error processing video list: {str(e)}")
    
    def extract_from_channel_dir(self, channel_dir):
        """Extract transcripts from all videos in a channel directory"""
        channel_path = Path(channel_dir)
        videos_list_file = channel_path / "videos-list.json"
        
        if videos_list_file.exists():
            # Set output to channel's transcript directory
            self.output_dir = channel_path / "transcripts"
            self.output_dir.mkdir(exist_ok=True)
            
            self.extract_from_video_list(videos_list_file)
        else:
            print(f"❌ Error: {videos_list_file} not found in channel directory")
    
    def sanitize_filename(self, filename):
        """Sanitize filename for saving"""
        # Remove/replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length and remove extra spaces
        return filename.strip()[:100]
    
    def print_stats(self):
        """Print processing statistics"""
        print(f"\n📈 Processing Statistics:")
        print(f"   📝 Total Processed: {self.stats['processed']}")
        print(f"   ✅ Successful: {self.stats['successful']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"   ⚠️  No Transcript: {self.stats['no_transcript']}")
        print(f"   📊 Success Rate: {(self.stats['successful']/self.stats['processed']*100):.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Extract YouTube video transcripts')
    parser.add_argument('input', nargs='?', help='Video ID, videos-list.json file, or channel directory')
    parser.add_argument('--batch', help='Process videos-list.json file')
    parser.add_argument('--channel-dir', help='Process all videos in channel directory')
    parser.add_argument('--output', default='transcripts', help='Output directory')
    parser.add_argument('--format', choices=['txt', 'json', 'both'], default='both', help='Output format')
    
    args = parser.parse_args()
    
    if not any([args.input, args.batch, args.channel_dir]):
        print("❌ Error: Please provide a video ID, --batch file, or --channel-dir")
        parser.print_help()
        return
    
    extractor = TranscriptExtractor(args.output)
    
    try:
        if args.channel_dir:
            extractor.extract_from_channel_dir(args.channel_dir)
        elif args.batch:
            extractor.extract_from_video_list(args.batch)
        elif args.input:
            if args.input.endswith('.json'):
                extractor.extract_from_video_list(args.input)
            elif os.path.isdir(args.input):
                extractor.extract_from_channel_dir(args.input)
            else:
                # Single video ID
                result = extractor.extract_transcript(args.input, save_files=True)
                if result['success']:
                    print(f"✅ Transcript extracted successfully")
                else:
                    print(f"❌ Failed to extract transcript: {result['error']}")
        
        extractor.print_stats()
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        extractor.print_stats()
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()