#!/usr/bin/env python3
"""
run_with_env.py - Run analysis with .env file support
"""

import os
import sys
import subprocess
from pathlib import Path

def load_env_file(env_path='.env'):
    """Load environment variables from .env file."""
    if not Path(env_path).exists():
        print(f"❌ No {env_path} file found")
        print("\nCreate a .env file with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        return False
    
    print(f"Loading environment from {env_path}...")
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                
                # Don't print the full key for security
                if key == 'OPENAI_API_KEY':
                    print(f"✓ {key} loaded (length: {len(value)})")
                else:
                    print(f"✓ {key} = {value}")
    
    return True

def main():
    """Run the improved analysis with .env support."""
    
    # Load .env file
    if load_env_file():
        print("\n✓ Environment loaded successfully")
    else:
        print("\n⚠️  Continuing without .env file")
    
    # Check if OPENAI_API_KEY is set
    if os.environ.get('OPENAI_API_KEY'):
        print("✓ OpenAI API key is available for LLM labeling")
    else:
        print("⚠️  No OpenAI API key - LLM labeling will be disabled")
    
    # Run the improved analysis
    print("\nRunning improved analysis...")
    print("="*60)
    
    # Pass command line arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else ['--auto_k']
    
    cmd = [sys.executable, 'run_analysis_improved.py'] + args
    
    # Run the command
    result = subprocess.run(cmd, env=os.environ)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())