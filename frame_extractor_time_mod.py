#!/usr/bin/env python3
"""
Video Frame Extractor with Clock Time Support

Extracts high-quality still frames from video files at specified timestamps
using a JSON configuration file. Supports single files, batch processing,
and extraction at specific clock times across multiple files.

Usage: 
  python frame_extractor_time.py time_config.json
  python frame_extractor_time.py --directory path/to/videos --timestamps "0:10,1:30,2:45"
  python frame_extractor_time.py --directory path/to/videos --clock-times "20250925151530,20250925152000"

Config file format for clock times:
{
    "video_directory": "path/to/videos",
    "clock_times": ["20250925151530", "20250925152000"],
    "output_directory": "frames",
    "output_format": "tiff",
    "recursive": false,
    "filename_pattern": "CAMERA_DATETIME_DATETIME"
}
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
import cv2
import glob

# Supported video file extensions
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.dav'}

def parse_filename(filename, pattern="CAMERA_DATETIME_DATETIME"):
    """Parse filename to extract camera name, start time, and end time."""
    if pattern == "CAMERA_DATETIME_DATETIME":
        stem = Path(filename).stem
        stem = re.sub(r'\(\d+\)$', '', stem)  # strip suffix like (1), (2)
        parts = stem.split('_')
        if len(parts) < 3:
            return None
        datetime_parts = []
        camera_parts = []
        for part in parts:
            if len(part) == 14 and part.isdigit():
                datetime_parts.append(part)
            else:
                if len(datetime_parts) < 2:
                    camera_parts.append(part)
        if len(datetime_parts) < 2:
            return None
        try:
            start_time = datetime.strptime(datetime_parts[0], "%Y%m%d%H%M%S")
            end_time = datetime.strptime(datetime_parts[1], "%Y%m%d%H%M%S")
            return {
                'camera': '_'.join(camera_parts),
                'start_time': start_time,
                'end_time': end_time,
                'filename': filename
            }
        except ValueError:
            return None
    return None

def parse_timestamp(timestamp_str):
    """Parse timestamp string to seconds."""
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

def parse_clock_time(clock_time_str):
    """Parse clock time string to datetime object (YYYYMMDDHHMMSS)."""
    return datetime.strptime(clock_time_str, "%Y%m%d%H%M%S")

def find_video_files(directory, recursive=False):
    """Find all video files in a directory."""
    video_files = []
    search_pattern = "**/*" if recursive else "*"
    for ext in VIDEO_EXTENSIONS:
        pattern = os.path.join(directory, search_pattern + ext)
        video_files.extend(glob.glob(pattern, recursive=recursive))
        pattern = os.path.join(directory, search_pattern + ext.upper())
        video_files.extend(glob.glob(pattern, recursive=recursive))
    return sorted(set(video_files))

# ----------------- MODIFIED SECTION ------------------

def extract_frame_at_clock_time_modified(video_info, target_time, output_dir, output_format):
    """Extract a frame from a video at a specific clock time and save as {video_file_name}.ext"""
    video_file = video_info['filename']
    start_time = video_info['start_time']
    offset_seconds = (target_time - start_time).total_seconds()

    print(f"\n--- Processing {Path(video_file).name} ---")
    print(f"Target time: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if offset_seconds < 0 or offset_seconds > duration:
        print(f"Error: Target time {offset_seconds:.2f}s is outside video duration ({duration:.2f}s)")
        cap.release()
        return False

    frame_number = int(offset_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not extract frame at {offset_seconds:.2f}s")
        cap.release()
        return False

    video_name = Path(video_file).stem
    output_filename = f"{video_name}.{output_format}"
    output_path = os.path.join(output_dir, output_filename)

    save_params = []
    if output_format == 'png':
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif output_format in ['jpg', 'jpeg']:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]

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
    """Extract frames from every video file (no camera grouping) at specific clock times."""
    target_times = [parse_clock_time(ct) for ct in clock_times]
    video_files = find_video_files(video_directory, recursive)
    if not video_files:
        print(f"No video files found in '{video_directory}'")
        return False

    videos = []
    for vf in video_files:
        info = parse_filename(vf, filename_pattern)
        if info:
            videos.append(info)
        else:
            print(f"Warning: Could not parse filename: {Path(vf).name}")

    if not videos:
        print("No videos could be parsed with the specified filename pattern")
        return False

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_success = 0
    for target_time in target_times:
        print(f"\n{'='*50}")
        print(f"Processing target time: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        time_success = 0
        for video_info in videos:
            if video_info['start_time'] <= target_time <= video_info['end_time']:
                if extract_frame_at_clock_time_modified(video_info, target_time, output_dir, output_format):
                    time_success += 1
                    total_success += 1
        print(f"\nTime {target_time.strftime('%Y-%m-%d %H:%M:%S')}: {time_success}/{len(videos)} videos processed successfully")

    print(f"\n{'='*60}")
    print(f"CLOCK TIME EXTRACTION COMPLETE: {total_success} total frames extracted")
    print(f"Processed {len(target_times)} time(s) across {len(videos)} videos")
    return total_success > 0

# -----------------------------------------------------

def extract_frames_from_video(video_file, timestamps, output_dir, output_format):
    """Extract frames from a single video file."""
    print(f"\n{'='*20} Processing: {Path(video_file).name} {'='*20}")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    video_name = Path(video_file).stem
    video_output_dir = os.path.join(output_dir, video_name)
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)
    success_count = 0
    for i, timestamp_str in enumerate(timestamps):
        try:
            ts_sec = parse_timestamp(timestamp_str)
            if ts_sec > duration:
                print(f"Warning: Timestamp {timestamp_str} exceeds video duration")
                continue
            frame_number = int(ts_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue
            output_filename = f"{video_name}_frame_{i+1:03d}.{output_format}"
            output_path = os.path.join(video_output_dir, output_filename)
            save_params = []
            if output_format == 'png':
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            elif output_format in ['jpg', 'jpeg']:
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
            if cv2.imwrite(output_path, frame, save_params):
                print(f"✓ Extracted frame {output_filename}")
                success_count += 1
        except Exception as e:
            print(f"Error: {e}")
    cap.release()
    return success_count

def extract_frames(config_path):
    """Extract frames based on configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    output_dir = config.get('output_directory', 'frames')
    output_format = config.get('output_format', 'png').lower()
    if 'clock_times' in config:
        return extract_frames_at_clock_times(
            config['video_directory'],
            config['clock_times'],
            output_dir,
            output_format,
            config.get('recursive', False),
            config.get('filename_pattern', 'CAMERA_DATETIME_DATETIME')
        )
    elif 'timestamps' in config:
        if 'video_file' in config:
            return extract_frames_from_video(
                config['video_file'],
                config['timestamps'],
                output_dir,
                output_format
            )
        else:
            video_files = find_video_files(config['video_directory'], config.get('recursive', False))
            total = 0
            for vf in video_files:
                total += extract_frames_from_video(vf, config['timestamps'], output_dir, output_format)
            return total > 0
    else:
        print("Config must contain 'clock_times' or 'timestamps'")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', nargs='?', help='JSON configuration file')
    parser.add_argument('-d', '--directory')
    parser.add_argument('-t', '--timestamps')
    parser.add_argument('-c', '--clock-times')
    parser.add_argument('-o', '--output', default='frames')
    parser.add_argument('-f', '--format', default='png', choices=['png','jpg','jpeg','tiff','bmp'])
    parser.add_argument('-r', '--recursive', action='store_true')
    args = parser.parse_args()
    if args.directory:
        if args.clock_times:
            extract_frames_at_clock_times(
                args.directory,
                [ct.strip() for ct in args.clock_times.split(',')],
                args.output,
                args.format,
                args.recursive
            )
        elif args.timestamps:
            ts = [t.strip() for t in args.timestamps.split(',')]
            for vf in find_video_files(args.directory, args.recursive):
                extract_frames_from_video(vf, ts, args.output, args.format)
    elif args.config_file:
        extract_frames(args.config_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
