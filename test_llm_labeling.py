#!/usr/bin/env python3
"""
test_llm_labeling.py

Test the LLM labeling functionality independently.
"""

import os
import openai
import json

def test_llm_labeling():
    """Test if LLM labeling is working correctly."""
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] No OPENAI_API_KEY found in environment variables")
        print("[INFO] Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print(f"[INFO] API key found (length: {len(api_key)})")
    
    # Initialize client
    try:
        client = openai.OpenAI(api_key=api_key)
        print("[INFO] OpenAI client initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}")
        return False
    
    # Test with sample criminal life events
    test_events = [
        "Parents divorced when he was 5 years old",
        "Physically abused by stepfather throughout childhood", 
        "Began torturing animals at age 12",
        "Dropped out of high school at 16",
        "First arrest for assault at age 19"
    ]
    
    prompt = (
        "You are an expert in criminal psychology and behavioral analysis.\n"
        "Given these representative life events of serial killers, identify\n"
        "the archetypal pattern or theme they represent. Be concise and specific.\n\n"
        "Life events:\n"
        + "\n".join(f"- {event}" for event in test_events)
        + "\n\nArchetypal theme:"
    )
    
    print("\n[INFO] Sending test prompt to LLM...")
    print("[INFO] Sample events:", test_events[:2], "...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in criminal psychology and behavioral analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        theme = response.choices[0].message.content.strip()
        print(f"\n[SUCCESS] LLM generated theme: '{theme}'")
        
        # Test cluster labeling format
        print("\n[INFO] Testing cluster labeling format...")
        clusters = [
            {
                "cluster_id": 0,
                "size": 100,
                "representative_samples": test_events[:3],
                "archetypal_theme": theme
            },
            {
                "cluster_id": 1, 
                "size": 50,
                "representative_samples": [
                    "Successful in school and sports",
                    "Popular with peers",
                    "Stable family environment"
                ],
                "archetypal_theme": "Well-adjusted youth development"
            }
        ]
        
        # Save test output
        with open("test_llm_output.json", "w") as f:
            json.dump(clusters, f, indent=2)
        
        print("[INFO] Test output saved to test_llm_output.json")
        print("\n[SUCCESS] LLM labeling is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] LLM API call failed: {e}")
        print("[INFO] Common issues:")
        print("  - Invalid API key")
        print("  - Insufficient credits")
        print("  - Network issues")
        return False


if __name__ == "__main__":
    print("="*60)
    print("TESTING LLM LABELING FUNCTIONALITY")
    print("="*60)
    
    success = test_llm_labeling()
    
    if success:
        print("\n✓ LLM labeling is ready to use!")
        print("\nTo run the full analysis with LLM labeling:")
        print("  python run_analysis_improved.py")
    else:
        print("\n✗ Please fix the issues above before running the analysis")