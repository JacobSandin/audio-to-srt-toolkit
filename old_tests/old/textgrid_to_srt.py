#!/usr/bin/env python3

import sys
import re
import os

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    # Ensure seconds is non-negative
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def parse_textgrid(textgrid_file):
    """Parse a TextGrid file and extract words with their timestamps"""
    with open(textgrid_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the word tier
    word_tier_match = re.search(r'item \[[0-9]+\]:\s*class = "IntervalTier"\s*name = "words"(.*?)item \[', content, re.DOTALL)
    if not word_tier_match:
        print("Could not find word tier in TextGrid file")
        return []
    
    word_tier = word_tier_match.group(1)
    
    # Extract intervals
    intervals = re.findall(r'intervals \[[0-9]+\]:\s*xmin = ([0-9\.]+)\s*xmax = ([0-9\.]+)\s*text = "([^"]*?)"', word_tier)
    
    words = []
    for xmin, xmax, text in intervals:
        # Skip empty, unknown, silence, and <eps> tokens
        if text.strip() and text.strip() != "<unk>" and text.strip() != "sp" and text.strip() != "<eps>":
            words.append({
                'start': float(xmin),
                'end': float(xmax),
                'text': text.strip()
            })
    
    return words

def words_to_subtitles(words, max_gap=0.5, max_duration=5.0, max_words=12):
    """Convert word-level timestamps to subtitle segments"""
    if not words:
        return []
    
    subtitles = []
    buffer = []
    start_time = None
    last_end = None
    
    for word in words:
        if start_time is None:
            start_time = word['start']
            last_end = word['end']
            buffer = [word]
            continue
        
        gap = word['start'] - last_end
        duration = word['end'] - start_time
        
        buffer.append(word)
        last_end = word['end']
        
        should_flush = (
            gap > max_gap or
            duration >= max_duration or
            len(buffer) >= max_words or
            word['text'][-1:] in ".?!"
        )
        
        if should_flush:
            text = " ".join(w['text'] for w in buffer)
            # Clean up the text - remove <eps> tokens
            text = re.sub(r'\s*<eps>\s*', ' ', text).strip()
            subtitles.append({
                'start': start_time,
                'end': last_end,
                'text': text
            })
            
            buffer = []
            start_time = None
    
    # Final flush
    if buffer:
        text = " ".join(w['text'] for w in buffer)
        # Clean up the text - remove <eps> tokens
        text = re.sub(r'\s*<eps>\s*', ' ', text).strip()
        subtitles.append({
            'start': start_time,
            'end': last_end,
            'text': text
        })
    
    return subtitles

def write_srt(subtitles, output_file):
    """Write subtitles to an SRT file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            start_str = format_timestamp(subtitle['start'])
            end_str = format_timestamp(subtitle['end'])
            
            f.write(f"{i}\n{start_str} --> {end_str}\n{subtitle['text']}\n\n")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.TextGrid output.srt")
        sys.exit(1)
    
    textgrid_file = sys.argv[1]
    srt_file = sys.argv[2]
    
    print(f"Parsing TextGrid file: {textgrid_file}")
    words = parse_textgrid(textgrid_file)
    print(f"Found {len(words)} words")
    
    print("Converting to subtitles...")
    subtitles = words_to_subtitles(words)
    print(f"Generated {len(subtitles)} subtitle segments")
    
    print(f"Writing SRT file: {srt_file}")
    write_srt(subtitles, srt_file)
    print("Done!")

if __name__ == "__main__":
    main()
