#!/usr/bin/env python3
"""
Video Frame Extractor with Clock Time Support

Extracts high-quality still frames from video files at specified timestamps
using a JSON configuration file. Supports single files, batch processing,
and extraction at specific clock times across multiple cameras.

Usage: 
  python frame_extractor_time.py time_config.json
  python frame_extractor_time.py --directory path/to/videos --timestamps "0:10,1:30,2:45"
  python frame_extractor_time.py --directory path/to/videos --clock-times "20250925151530,20250925152000"

Config file format for clock times:
{
    "video_directory": "path/to/videos",
    "clock_times": ["20250925151530", "20250925152000"],
    "output_directory": "frames",
    "output_format": "png",
    "recursive": false,
    "filename_pattern": "CAMERA_DATETIME_DATETIME"
}

Filename pattern examples:
- "CAMERA_DATETIME_DATETIME": N910A6_ch1_main_20250925151500_20250925153742.avi
- Custom patterns can be added as needed
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
import cv2
import glob
from collections import defaultdict


# Supported video file extensions
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.dav'}


def parse_filename(filename, pattern="CAMERA_DATETIME_DATETIME"):
    """
    Parse filename to extract camera name, start time, and end time.
    
    Args:
        filename: Video filename (handles suffixes like (1), (2), etc.)
        pattern: Pattern type for parsing
        
    Returns:
        dict with 'camera', 'start_time', 'end_time' as datetime objects
    """
    if pattern == "CAMERA_DATETIME_DATETIME":
        # Pattern: CAMERA_DATETIME_DATETIME.ext or CAMERA_DATETIME_DATETIME(N).ext
        # Example: N910A6_ch1_main_20250925151500_20250925153742.avi
        # Example: N910A6_ch6_main_20250925133217_20250925133412(1).dav
        
        stem = Path(filename).stem
        
        # Remove any parenthetical suffix like (1), (2), etc.
        stem = re.sub(r'\(\d+\)$', '', stem)
        
        parts = stem.split('_')
        
        if len(parts) < 3:
            return None
            
        # Find the datetime parts (14 digits each)
        datetime_parts = []
        camera_parts = []
        
        for part in parts:
            if len(part) == 14 and part.isdigit():
                datetime_parts.append(part)
            else:
                if len(datetime_parts) < 2:  # Still looking for datetime parts
                    camera_parts.append(part)
        
        if len(datetime_parts) < 2:
            return None
            
        camera_name = '_'.join(camera_parts)
        start_time_str = datetime_parts[0]
        end_time_str = datetime_parts[1]
        
        try:
            start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")
            end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M%S")
            
            return {
                'camera': camera_name,
                'start_time': start_time,
                'end_time': end_time,
                'filename': filename
            }
        except ValueError:
            return None
    
    return None


def parse_timestamp(timestamp_str):
    """
    Parse timestamp string to seconds.
    Supports formats: "1:30", "0:10.5", "90", "90.5"
    """
    try:
        if ':' in timestamp_str:
            parts = timestamp_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            return float(timestamp_str)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}") from e


def parse_clock_time(clock_time_str):
    """
    Parse clock time string to datetime object.
    Supports format: "20250925151530" (YYYYMMDDHHMMSS)
    """
    try:
        return datetime.strptime(clock_time_str, "%Y%m%d%H%M%S")
    except ValueError as e:
        raise ValueError(f"Invalid clock time format: {clock_time_str}. Use YYYYMMDDHHMMSS format.") from e


def find_video_files(directory, recursive=False):
    """Find all video files in a directory."""
    video_files = []
    search_pattern = "**/*" if recursive else "*"
    
    for ext in VIDEO_EXTENSIONS:
        pattern = os.path.join(directory, search_pattern + ext)
        files = glob.glob(pattern, recursive=recursive)
        video_files.extend(files)
        
        # Also search for uppercase extensions
        pattern = os.path.join(directory, search_pattern + ext.upper())
        files = glob.glob(pattern, recursive=recursive)
        video_files.extend(files)
    
    return sorted(set(video_files))  # Remove duplicates and sort


def organize_videos_by_camera(video_files, filename_pattern="CAMERA_DATETIME_DATETIME"):
    """
    Organize video files by camera and time ranges.
    
    Returns:
        dict: {camera_name: [video_info_dicts]}
    """
    cameras = defaultdict(list)
    
    for video_file in video_files:
        info = parse_filename(video_file, filename_pattern)
        if info:
            cameras[info['camera']].append(info)
        else:
            print(f"Warning: Could not parse filename: {Path(video_file).name}")
    
    # Sort videos by start time for each camera
    for camera in cameras:
        cameras[camera].sort(key=lambda x: x['start_time'])
    
    return cameras


def find_video_for_clock_time(camera_videos, target_time):
    """
    Find the video file that contains the target clock time for a specific camera.
    
    Args:
        camera_videos: List of video info dicts for one camera
        target_time: datetime object of target time
        
    Returns:
        dict: Video info dict if found, None otherwise
    """
    for video_info in camera_videos:
        if video_info['start_time'] <= target_time <= video_info['end_time']:
            return video_info
    return None


def extract_frame_at_clock_time(video_info, target_time, output_dir, output_format):
    """
    Extract a frame from a video at a specific clock time.
    
    Args:
        video_info: Dict with video information including filename and times
        target_time: datetime object of target time
        output_dir: Output directory
        output_format: Output format (png, jpg, etc.)
        
    Returns:
        bool: Success status
    """
    video_file = video_info['filename']
    camera = video_info['camera']
    start_time = video_info['start_time']
    
    # Calculate offset from video start
    time_offset = target_time - start_time
    offset_seconds = time_offset.total_seconds()
    
    print(f"\n--- Processing {camera} ---")
    print(f"Video: {Path(video_file).name}")
    print(f"Target time: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Video start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Offset: {offset_seconds:.2f} seconds")
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"FPS: {fps:.2f}, Duration: {duration:.2f}s")
    
    if offset_seconds > duration:
        print(f"Error: Target time is beyond video duration ({duration:.2f}s)")
        cap.release()
        return False
    
    if offset_seconds < 0:
        print(f"Error: Target time is before video start")
        cap.release()
        return False
    
    # Calculate frame number
    frame_number = int(offset_seconds * fps)
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not extract frame at offset {offset_seconds:.2f}s")
        cap.release()
        return False
    
    # Create output filename
    time_str = target_time.strftime('%Y%m%d_%H%M%S')
    output_filename = f"{camera}_{time_str}.{output_format}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Set compression parameters for maximum quality
    if output_format == 'png':
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif output_format in ['jpg', 'jpeg']:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif output_format == 'tiff':
        save_params = []
    elif output_format == 'bmp':
        save_params = []
    else:
        save_params = []
    
    # Save frame
    if cv2.imwrite(output_path, frame, save_params):
        print(f"✓ Extracted frame → {output_filename}")
        cap.release()
        return True
    else:
        print(f"✗ Failed to save frame")
        cap.release()
        return False


def extract_frames_at_clock_times(video_directory, clock_times, output_dir, 
                                output_format, recursive=False, 
                                filename_pattern="CAMERA_DATETIME_DATETIME"):
    """
    Extract frames from multiple cameras at specific clock times.
    """
    # Parse clock times
    target_times = []
    for clock_time_str in clock_times:
        try:
            target_time = parse_clock_time(clock_time_str)
            target_times.append(target_time)
        except ValueError as e:
            print(f"Error parsing clock time '{clock_time_str}': {e}")
            return False
    
    # Find all video files
    video_files = find_video_files(video_directory, recursive)
    if not video_files:
        print(f"No video files found in '{video_directory}'")
        return False
    
    print(f"Found {len(video_files)} video file(s)")
    
    # Organize videos by camera
    cameras = organize_videos_by_camera(video_files, filename_pattern)
    
    if not cameras:
        print("No videos could be parsed with the specified filename pattern")
        return False
    
    print(f"Found {len(cameras)} camera(s): {', '.join(cameras.keys())}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each target time
    total_success = 0
    for target_time in target_times:
        print(f"\n{'='*50}")
        print(f"Processing target time: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        
        time_success = 0
        for camera, camera_videos in cameras.items():
            # Find video that contains this time
            video_info = find_video_for_clock_time(camera_videos, target_time)
            
            if video_info:
                if extract_frame_at_clock_time(video_info, target_time, output_dir, output_format):
                    time_success += 1
                    total_success += 1
            else:
                print(f"\n--- {camera} ---")
                print(f"No video found containing time {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Show available time ranges for this camera
                print("Available time ranges for this camera:")
                for video in camera_videos:
                    start_str = video['start_time'].strftime('%Y-%m-%d %H:%M:%S')
                    end_str = video['end_time'].strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  {Path(video['filename']).name}: {start_str} - {end_str}")
        
        print(f"\nTime {target_time.strftime('%Y-%m-%d %H:%M:%S')}: {time_success}/{len(cameras)} cameras processed successfully")
    
    print(f"\n{'='*60}")
    print(f"CLOCK TIME EXTRACTION COMPLETE: {total_success} total frames extracted")
    print(f"Processed {len(target_times)} time(s) across {len(cameras)} camera(s)")
    
    return total_success > 0


def extract_frames_from_video(video_file, timestamps, output_dir, output_format):
    """Extract frames from a single video file."""
    print(f"\n{'='*20} Processing: {Path(video_file).name} {'='*20}")
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return 0
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"FPS: {fps:.2f}, Duration: {duration:.2f}s, Total frames: {total_frames}")
    
    # Create subdirectory for this video
    video_name = Path(video_file).stem
    video_output_dir = os.path.join(output_dir, video_name)
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each timestamp
    success_count = 0
    for i, timestamp_str in enumerate(timestamps):
        try:
            # Parse timestamp
            timestamp_seconds = parse_timestamp(timestamp_str)
            
            if timestamp_seconds > duration:
                print(f"Warning: Timestamp {timestamp_str} ({timestamp_seconds:.2f}s) exceeds video duration ({duration:.2f}s)")
                continue
            
            # Calculate frame number
            frame_number = int(timestamp_seconds * fps)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not extract frame at {timestamp_str}")
                continue
            
            # Generate output filename
            output_filename = f"{video_name}_frame_{i+1:03d}_{timestamp_str.replace(':', 'm').replace('.', 's')}.{output_format}"
            output_path = os.path.join(video_output_dir, output_filename)
            
            # Set compression parameters for maximum quality
            if output_format == 'png':
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            elif output_format in ['jpg', 'jpeg']:
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
            elif output_format == 'tiff':
                save_params = []
            elif output_format == 'bmp':
                save_params = []
            else:
                save_params = []
            
            # Save frame
            if cv2.imwrite(output_path, frame, save_params):
                print(f"✓ Extracted frame at {timestamp_str} → {output_filename}")
                success_count += 1
            else:
                print(f"✗ Failed to save frame at {timestamp_str}")
                
        except ValueError as e:
            print(f"✗ Error processing timestamp '{timestamp_str}': {e}")
        except Exception as e:
            print(f"✗ Unexpected error processing timestamp '{timestamp_str}': {e}")
    
    cap.release()
    print(f"Video complete: {success_count}/{len(timestamps)} frames extracted")
    return success_count


def extract_frames(config_path):
    """Extract frames based on configuration file."""
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return False
    
    # Check processing mode
    has_single_file = 'video_file' in config
    has_directory = 'video_directory' in config
    has_clock_times = 'clock_times' in config
    has_timestamps = 'timestamps' in config
    
    if not has_single_file and not has_directory:
        print("Error: Config file must contain either 'video_file' or 'video_directory'.")
        return False
    
    if has_single_file and has_directory:
        print("Error: Config file cannot contain both 'video_file' and 'video_directory'.")
        return False
    
    if has_clock_times and has_timestamps:
        print("Error: Config file cannot contain both 'clock_times' and 'timestamps'.")
        return False
    
    if not has_clock_times and not has_timestamps:
        print("Error: Config file must contain either 'clock_times' or 'timestamps'.")
        return False
    
    output_dir = config.get('output_directory', 'frames')
    output_format = config.get('output_format', 'png').lower()
    
    # Validate output format
    valid_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    if output_format not in valid_formats:
        print(f"Error: Unsupported output format '{output_format}'. Use: {', '.join(valid_formats)}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    total_success = 0
    
    if has_clock_times:
        # Clock time processing (only works with directory mode)
        if not has_directory:
            print("Error: 'clock_times' can only be used with 'video_directory'.")
            return False
            
        video_directory = config['video_directory']
        clock_times = config['clock_times']
        recursive = config.get('recursive', False)
        filename_pattern = config.get('filename_pattern', 'CAMERA_DATETIME_DATETIME')
        
        print(f"Clock time processing mode")
        print(f"Video directory: {video_directory}")
        print(f"Output directory: {output_dir}")
        print(f"Output format: {output_format.upper()}")
        print(f"Clock times: {', '.join(clock_times)}")
        print(f"Filename pattern: {filename_pattern}")
        print(f"Recursive: {recursive}")
        
        return extract_frames_at_clock_times(
            video_directory, clock_times, output_dir, output_format, 
            recursive, filename_pattern
        )
    
    else:
        # Original timestamp processing
        timestamps = config['timestamps']
        
        print(f"Output directory: {output_dir}")
        print(f"Output format: {output_format.upper()}")
        print(f"Timestamps: {', '.join(timestamps)}")
        
        if has_single_file:
            # Single file processing
            video_file = config['video_file']
            if not os.path.exists(video_file):
                print(f"Error: Video file '{video_file}' not found.")
                return False
            
            total_success = extract_frames_from_video(video_file, timestamps, output_dir, output_format)
            
        else:
            # Directory processing with timestamps
            video_directory = config['video_directory']
            recursive = config.get('recursive', False)
            
            if not os.path.exists(video_directory):
                print(f"Error: Video directory '{video_directory}' not found.")
                return False
            
            if not os.path.isdir(video_directory):
                print(f"Error: '{video_directory}' is not a directory.")
                return False
            
            # Find all video files
            video_files = find_video_files(video_directory, recursive)
            
            if not video_files:
                print(f"No video files found in '{video_directory}'")
                print(f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
                return False
            
            print(f"Found {len(video_files)} video file(s) in '{video_directory}':")
            for vf in video_files:
                print(f"  - {Path(vf).name}")
            
            # Process each video file
            for video_file in video_files:
                success_count = extract_frames_from_video(video_file, timestamps, output_dir, output_format)
                total_success += success_count
    
    print("\n" + "="*60)
    print(f"EXTRACTION COMPLETE: {total_success} total frames extracted successfully")
    
    return total_success > 0


def create_sample_config(filename="config.json", config_type="single"):
    """Create a sample configuration file."""
    if config_type == "clock":
        sample_config = {
            "video_directory": "path/to/video/folder",
            "clock_times": ["20250925151530", "20250925152000", "20250925152500"],
            "output_directory": "frames",
            "output_format": "png",
            "recursive": False,
            "filename_pattern": "CAMERA_DATETIME_DATETIME"
        }
        config_desc = "clock time extraction"
    elif config_type == "batch":
        sample_config = {
            "video_directory": "path/to/video/folder",
            "timestamps": ["0:10", "1:30", "2:45.5"],
            "output_directory": "frames",
            "output_format": "png",
            "recursive": False
        }
        config_desc = "batch processing"
    else:
        sample_config = {
            "video_file": "example.avi",
            "timestamps": ["0:10", "1:30", "2:45.5"],
            "output_directory": "frames",
            "output_format": "png"
        }
        config_desc = "single file"
    
    with open(filename, 'w') as f:
        json.dump(sample_config, f, indent=4)
    
    print(f"Sample configuration file for {config_desc} created: {filename}")
    print("Edit this file with your video path(s) and desired times.")


def process_directory_direct(video_directory, timestamps_str=None, clock_times_str=None,
                           output_dir="frames", output_format="png", recursive=False):
    """Process directory directly from command line arguments."""
    
    if timestamps_str and clock_times_str:
        print("Error: Cannot specify both --timestamps and --clock-times")
        return False
    
    if not timestamps_str and not clock_times_str:
        print("Error: Must specify either --timestamps or --clock-times")
        return False
    
    # Validate directory
    if not os.path.exists(video_directory):
        print(f"Error: Directory '{video_directory}' not found.")
        return False
    
    if not os.path.isdir(video_directory):
        print(f"Error: '{video_directory}' is not a directory.")
        return False
    
    # Validate output format
    valid_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    if output_format.lower() not in valid_formats:
        print(f"Error: Unsupported output format '{output_format}'. Use: {', '.join(valid_formats)}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if clock_times_str:
        # Clock time processing
        clock_times = [ct.strip() for ct in clock_times_str.split(',')]
        print(f"Clock time processing mode")
        print(f"Clock times: {', '.join(clock_times)}")
        
        return extract_frames_at_clock_times(
            video_directory, clock_times, output_dir, output_format.lower(), recursive
        )
    
    else:
        # Original timestamp processing
        timestamps = [ts.strip() for ts in timestamps_str.split(',')]
        
        # Find video files
        video_files = find_video_files(video_directory, recursive)
        
        if not video_files:
            print(f"No video files found in '{video_directory}'")
            print(f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
            return False
        
        print(f"Processing {len(video_files)} video file(s) from '{video_directory}'")
        print(f"Timestamps: {', '.join(timestamps)}")
        print(f"Output directory: {output_dir}")
        print(f"Output format: {output_format.upper()}")
        print(f"Recursive: {recursive}")
        
        # Process each video file
        total_success = 0
        for video_file in video_files:
            success_count = extract_frames_from_video(video_file, timestamps, output_dir, output_format.lower())
            total_success += success_count
        
        print("\n" + "="*60)
        print(f"BATCH PROCESSING COMPLETE: {total_success} total frames extracted successfully")
        
        return total_success > 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract high-quality frames from video files at specified timestamps or clock times",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file via config
  python frame_extractor_time.py time_config.json
  
  # Batch process directory with relative timestamps
  python frame_extractor_time.py --directory /path/to/videos --timestamps "0:10,1:30,2:45"
  
  # Extract frames at specific clock times from multiple cameras
  python frame_extractor_time.py --directory /path/to/videos --clock-times "20250925151530,20250925152000"
  
  # Create sample configs
  python frame_extractor_time.py --sample-config
  python frame_extractor_time.py --sample-config-batch
  python frame_extractor_time.py --sample-config-clock
  
Clock time config format:
{
    "video_directory": "path/to/videos",
    "clock_times": ["20250925151530", "20250925152000"],
    "output_directory": "frames",
    "output_format": "png",
    "recursive": false,
    "filename_pattern": "CAMERA_DATETIME_DATETIME"
}

Supported filename patterns:
  CAMERA_DATETIME_DATETIME: N910A6_ch1_main_20250925151500_20250925153742.avi

Supported video formats: """ + ", ".join(sorted(VIDEO_EXTENSIONS)) + """

Clock time format: YYYYMMDDHHMMSS
  20250925151530 = September 25, 2025 at 15:15:30

Timestamp formats (relative to video start):
  "1:30"     - 1 minute 30 seconds
  "0:10.5"   - 10.5 seconds  
  "90"       - 90 seconds
  "1:2:30"   - 1 hour 2 minutes 30 seconds
        """
    )
    
    parser.add_argument(
        'config_file',
        nargs='?',
        help='JSON configuration file path (not used with --directory)'
    )
    
    parser.add_argument(
        '-d', '--directory',
        help='Directory containing video files to process'
    )
    
    parser.add_argument(
        '-t', '--timestamps',
        help='Comma-separated relative timestamps (e.g., "0:10,1:30,2:45")'
    )
    
    parser.add_argument(
        '-c', '--clock-times',
        help='Comma-separated absolute clock times (e.g., "20250925151530,20250925152000")'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='frames',
        help='Output directory (default: frames)'
    )
    
    parser.add_argument(
        '-f', '--format',
        default='png',
        choices=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help='Output format (default: png)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Search for video files recursively in subdirectories'
    )
    
    parser.add_argument(
        '--sample-config',
        action='store_true',
        help='Create a sample configuration file for single file processing'
    )
    
    parser.add_argument(
        '--sample-config-batch',
        action='store_true', 
        help='Create a sample configuration file for batch processing'
    )
    
    parser.add_argument(
        '--sample-config-clock',
        action='store_true',
        help='Create a sample configuration file for clock time extraction'
    )
    
    args = parser.parse_args()
    
    if args.sample_config:
        create_sample_config("config.json", "single")
        return
    
    if args.sample_config_batch:
        create_sample_config("config_batch.json", "batch")
        return
        
    if args.sample_config_clock:
        create_sample_config("config_clock.json", "clock")
        return
    
    # Directory mode
    if args.directory:
        if not args.timestamps and not args.clock_times:
            print("Error: --timestamps or --clock-times is required when using --directory")
            return
        if args.config_file:
            print("Warning: config_file argument ignored when using --directory")
        
        success = process_directory_direct(
            args.directory, 
            args.timestamps, 
            args.clock_times,
            args.output, 
            args.format,
            args.recursive
        )
        sys.exit(0 if success else 1)
    
    # Config file mode
    if not args.config_file:
        parser.print_help()
        return
    
    success = extract_frames(args.config_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()