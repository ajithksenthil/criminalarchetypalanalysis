#!/usr/bin/env python3
"""
run_with_progress.py - Run analysis with progress monitoring
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

def load_env_file(env_path='.env'):
    """Load environment variables from .env file."""
    if not Path(env_path).exists():
        return False
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")
    return True

def monitor_output(process, total_events=2617):
    """Monitor process output and show progress."""
    events_processed = 0
    start_time = time.time()
    
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            
            # Track progress
            if "[DEBUG] Processing event" in line:
                events_processed += 1
                elapsed = time.time() - start_time
                rate = events_processed / elapsed if elapsed > 0 else 0
                eta = (total_events - events_processed) / rate if rate > 0 else 0
                
                if events_processed % 50 == 0:
                    print(f"\n⏱️  Progress: {events_processed}/{total_events} events "
                          f"({events_processed/total_events*100:.1f}%) "
                          f"- Rate: {rate:.1f} events/sec "
                          f"- ETA: {eta/60:.1f} minutes\n")

def main():
    """Run analysis with progress monitoring."""
    
    # Load .env
    if load_env_file():
        print("✓ Environment loaded from .env")
        if os.environ.get('OPENAI_API_KEY'):
            print("✓ OpenAI API key detected - LLM labeling enabled")
            print("⚠️  Note: LLM processing takes ~30-45 minutes for full dataset\n")
        else:
            print("⚠️  No OpenAI API key - LLM labeling disabled\n")
    
    # Suggest faster alternatives
    print("Options for faster processing:")
    print("1. Skip LLM processing: Add --no_llm flag")
    print("2. Use TF-IDF embeddings: Add --use_tfidf flag")
    print("3. Process subset: Add --max_events 500")
    print("\nPress Ctrl+C to cancel and choose different options\n")
    
    # Wait a moment for user to read
    time.sleep(3)
    
    # Build command
    args = sys.argv[1:] if len(sys.argv) > 1 else ['--auto_k']
    cmd = [sys.executable, 'run_analysis_improved.py'] + args
    
    print(f"Running: {' '.join(cmd)}")
    print("="*60)
    
    # Run with output monitoring
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=os.environ
    )
    
    try:
        # Monitor in thread
        monitor_thread = threading.Thread(target=monitor_output, args=(process,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for completion
        return_code = process.wait()
        monitor_thread.join()
        
        if return_code == 0:
            print("\n✓ Analysis completed successfully!")
        else:
            print(f"\n❌ Analysis failed with code {return_code}")
            
        return return_code
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        process.terminate()
        return 1

if __name__ == "__main__":
    sys.exit(main())