#!/usr/bin/env python3
"""
test_lexical_bias_integration.py

Test script to verify lexical bias reduction is working properly
and create a corrected version that processes text throughout the pipeline.
"""

import json
from pathlib import Path
from lexical_bias_processor import LexicalBiasProcessor

def test_current_output():
    """Test the current output to see if lexical bias reduction worked."""
    
    print("TESTING LEXICAL BIAS REDUCTION IN CURRENT OUTPUT")
    print("=" * 60)
    
    # Load cluster info from the lexical bias reduced output
    cluster_file = "output_lexical_bias_reduced/clustering/cluster_info.json"
    
    if not Path(cluster_file).exists():
        print(f"❌ File not found: {cluster_file}")
        return
    
    with open(cluster_file, 'r') as f:
        cluster_info = json.load(f)
    
    print(f"\n📊 Analyzing {len(cluster_info)} clusters...")
    
    processor = LexicalBiasProcessor()
    
    bias_found = False
    
    for cluster in cluster_info:
        cluster_id = cluster['cluster_id']
        samples = cluster['representative_samples']
        
        print(f"\n🔍 Cluster {cluster_id}:")
        
        for i, sample in enumerate(samples[:2]):  # Check first 2 samples
            print(f"  Sample {i+1}:")
            print(f"    Original: {sample[:100]}...")
            
            # Process with bias reduction
            processed = processor.process_text(sample)
            print(f"    Should be: {processed[:100]}...")
            
            # Check if bias reduction is needed
            if sample != processed:
                bias_found = True
                print(f"    ❌ BIAS DETECTED: Names/locations not removed")
            else:
                print(f"    ✅ Clean: No bias detected")
    
    if bias_found:
        print(f"\n❌ CONCLUSION: Lexical bias reduction is NOT working properly")
        print(f"   The representative samples still contain specific names and locations")
        print(f"   This means the processed text is not being used for cluster analysis")
    else:
        print(f"\n✅ CONCLUSION: Lexical bias reduction is working correctly")

def create_corrected_analysis():
    """Create a corrected analysis with proper lexical bias reduction."""
    
    print(f"\n" + "=" * 60)
    print("CREATING CORRECTED ANALYSIS")
    print("=" * 60)
    
    # Load original data
    try:
        with open("output_lexical_bias_reduced/criminal_sequences.json", 'r') as f:
            sequences = json.load(f)
        
        print(f"📊 Loaded {len(sequences)} criminal sequences")
        
        # Extract all events
        all_events = []
        for criminal_id, events in sequences.items():
            all_events.extend(events)
        
        print(f"📊 Total events: {len(all_events)}")
        
        # Apply lexical bias reduction
        processor = LexicalBiasProcessor()
        processed_events = processor.process_event_list(all_events)
        
        # Analyze bias reduction effectiveness
        analysis = processor.analyze_bias_reduction(all_events[:100], processed_events[:100])
        
        print(f"\n📈 BIAS REDUCTION ANALYSIS:")
        print(f"   Original entities: {analysis.get('original_entity_count', 'N/A')}")
        print(f"   Processed entities: {analysis.get('processed_entity_count', 'N/A')}")
        print(f"   Reduction rate: {analysis.get('reduction_rate', 0):.1%}")
        
        # Show sample transformations
        print(f"\n📝 SAMPLE TRANSFORMATIONS:")
        samples = analysis.get('sample_replacements', [])
        for i, sample in enumerate(samples[:3]):
            print(f"\n   Example {i+1}:")
            print(f"   Before: {sample['original']}")
            print(f"   After:  {sample['processed']}")
        
        # Save processed events for manual inspection
        output_file = "processed_events_sample.json"
        sample_data = {
            "original_events": all_events[:20],
            "processed_events": processed_events[:20],
            "bias_analysis": analysis
        }
        
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"\n💾 Sample processed events saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")

def show_recommendations():
    """Show recommendations for proper lexical bias integration."""
    
    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS FOR PROPER LEXICAL BIAS REDUCTION")
    print("=" * 60)
    
    recommendations = """
🎯 ISSUE IDENTIFIED:
   The lexical bias reduction is applied to embeddings but not to the text
   used for representative sample selection and cluster analysis.

🔧 SOLUTION NEEDED:
   1. Process text BEFORE embedding generation
   2. Store processed text alongside original text
   3. Use processed text for:
      - Embedding generation
      - Representative sample selection
      - Cluster labeling
      - All text analysis

💡 IMPLEMENTATION:
   1. Modify data loading to apply lexical bias reduction early
   2. Keep both original and processed versions
   3. Use processed version for analysis, original for reference

🚀 COMMAND TO RUN CORRECTED ANALYSIS:
   python run_modular_analysis.py \\
     --type1_dir=type1csvs \\
     --type2_csv=type2csvs \\
     --output_dir=output_corrected_bias_reduction \\
     --auto_k --match_only --no_llm \\
     --embedding_model=all-mpnet-base-v2 \\
     --use_lexical_bias_reduction \\
     --n_clusters=5

📊 EXPECTED RESULTS:
   - Representative samples should contain [PERSON], [LOCATION], etc.
   - No specific names or places in cluster descriptions
   - Focus on behavioral patterns rather than individual cases
   - Improved clustering quality due to reduced lexical bias
"""
    
    print(recommendations)

def main():
    """Main function."""
    print("🔬 LEXICAL BIAS REDUCTION TESTING & VALIDATION")
    print("Testing whether lexical bias reduction is working properly...")
    
    # Test current output
    test_current_output()
    
    # Create corrected analysis
    create_corrected_analysis()
    
    # Show recommendations
    show_recommendations()
    
    print(f"\n✅ Testing complete!")
    print(f"Check the analysis above to see if lexical bias reduction is working.")

if __name__ == "__main__":
    main()
