#!/usr/bin/env python3
"""
integrated_prototype_system.py

Complete integrated system that combines:
1. Entity replacement (remove names, locations)
2. LLM lexical variations (reduce word choice bias)  
3. Prototype embeddings (average variations)
4. State-of-the-art embeddings (OpenAI or Sentence-BERT)
"""

import numpy as np
from typing import List, Dict, Any
import openai
import os
from dotenv import load_dotenv
import time
from lexical_bias_processor import LexicalBiasProcessor

load_dotenv()

class IntegratedPrototypeProcessor:
    """Complete prototype processing with entity replacement + LLM variations."""
    
    def __init__(self, embedding_model: str = "all-mpnet-base-v2", 
                 use_openai: bool = False, num_variations: int = 3):
        """
        Initialize the integrated prototype processor.
        
        Args:
            embedding_model: Embedding model to use
            use_openai: Whether to use OpenAI embeddings
            num_variations: Number of lexical variations per event
        """
        self.embedding_model = embedding_model
        self.use_openai = use_openai
        self.num_variations = num_variations
        
        # Initialize components
        self.bias_processor = LexicalBiasProcessor()
        
        if use_openai:
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print(f"[INFO] Using OpenAI embeddings: {embedding_model}")
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embedding_model)
            self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # For variations
            print(f"[INFO] Using Sentence-BERT embeddings: {embedding_model}")
    
    def generate_lexical_variations(self, processed_text: str) -> List[str]:
        """
        Generate lexical variations of entity-replaced text using LLM.
        
        Args:
            processed_text: Text with entities already replaced
            
        Returns:
            List of lexical variations
        """
        prompt = f"""Generate {self.num_variations} alternative versions of this criminal life event description.
The text already has entities replaced with generic labels like [PERSON], [LOCATION], etc.
Keep these labels exactly as they are, but vary the vocabulary, sentence structure, and phrasing.
Focus on different ways to express the same behavioral pattern.

Original: {processed_text}

Alternative versions (keep [PERSON], [LOCATION] etc. unchanged):"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective for this task
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Parse variations
            variations = []
            for line in reply.split('\n'):
                line = line.strip()
                # Remove numbering/bullets
                cleaned = line.lstrip('0123456789.-) ').strip()
                if cleaned and len(cleaned) > 10:
                    variations.append(cleaned)
            
            return variations[:self.num_variations]
            
        except Exception as e:
            print(f"[WARNING] Could not generate variations for: {processed_text[:50]}... Error: {e}")
            return [processed_text]  # Fallback to original
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if self.use_openai:
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.embedding_model
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                print(f"[WARNING] OpenAI embedding failed: {e}")
                # Fallback to zeros
                return np.zeros(3072 if "large" in self.embedding_model else 1536)
        else:
            return self.model.encode([text])[0]
    
    def get_prototype_embedding(self, original_text: str) -> Dict[str, Any]:
        """
        Get prototype embedding for a single event.
        
        Args:
            original_text: Original life event description
            
        Returns:
            Dictionary with prototype embedding and processing info
        """
        # Step 1: Entity replacement
        processed_text = self.bias_processor.process_text(original_text)
        
        # Step 2: Generate lexical variations
        variations = self.generate_lexical_variations(processed_text)
        
        # Step 3: Generate embeddings for all versions
        all_versions = [processed_text] + variations
        embeddings = []
        
        for version in all_versions:
            embedding = self.generate_embedding(version)
            embeddings.append(embedding)
        
        # Step 4: Create prototype (average) embedding
        embeddings_array = np.array(embeddings)
        prototype_embedding = np.mean(embeddings_array, axis=0)
        
        return {
            'original_text': original_text,
            'processed_text': processed_text,
            'variations': variations,
            'prototype_embedding': prototype_embedding,
            'num_variations': len(variations)
        }
    
    def process_event_list(self, events: List[str]) -> Dict[str, Any]:
        """
        Process a list of criminal life events with full prototype processing.
        
        Args:
            events: List of life event descriptions
            
        Returns:
            Dictionary with embeddings and processed texts
        """
        print(f"[INFO] Processing {len(events)} events with integrated prototype system")
        print(f"[INFO] Entity replacement + {self.num_variations} LLM variations + prototype embeddings")
        
        prototype_embeddings = []
        processed_texts = []
        processing_info = []
        
        for i, event in enumerate(events):
            if i % 50 == 0:
                print(f"  Processing event {i+1}/{len(events)}")
            
            result = self.get_prototype_embedding(event)
            
            prototype_embeddings.append(result['prototype_embedding'])
            processed_texts.append(result['processed_text'])
            processing_info.append({
                'original': result['original_text'],
                'processed': result['processed_text'],
                'variations': result['variations']
            })
            
            # Rate limiting for API calls
            if self.use_openai or self.client:
                time.sleep(0.1)
        
        return {
            'embeddings': np.array(prototype_embeddings),
            'processed_texts': processed_texts,
            'processing_info': processing_info,
            'method': f"Integrated Prototype ({self.embedding_model})"
        }

def test_integrated_prototype_system():
    """Test the integrated prototype system."""
    
    print("TESTING INTEGRATED PROTOTYPE SYSTEM")
    print("=" * 50)
    
    # Sample criminal events
    sample_events = [
        "Charles Albright was born in Amarillo, Texas on August 10, 1933",
        "Defense expert Samuel J. Palenik testified that hair samples may not belong to Albright",
        "Jury convicts Albright on murder of Shirley Williams in Dallas County Court",
        "Born in West Orange, New Jersey as last of 8 children",
        "Served 3 years in Folsom State Prison for armed robbery"
    ]
    
    # Test with Sentence-BERT (no API costs)
    print("\n🧪 Testing with Sentence-BERT...")
    processor = IntegratedPrototypeProcessor(
        embedding_model="all-MiniLM-L6-v2",
        use_openai=False,
        num_variations=2
    )
    
    # Process one event to show the full pipeline
    print(f"\n📝 PROCESSING SAMPLE EVENT:")
    sample_event = sample_events[0]
    print(f"Original: {sample_event}")
    
    result = processor.get_prototype_embedding(sample_event)
    
    print(f"\n🔄 STEP 1 - Entity Replacement:")
    print(f"Processed: {result['processed_text']}")
    
    print(f"\n🔄 STEP 2 - LLM Variations:")
    for i, variation in enumerate(result['variations'], 1):
        print(f"Variation {i}: {variation}")
    
    print(f"\n🔄 STEP 3 - Prototype Embedding:")
    print(f"Embedding shape: {result['prototype_embedding'].shape}")
    print(f"Embedding preview: {result['prototype_embedding'][:5]}...")
    
    print(f"\n✅ INTEGRATED PROTOTYPE PROCESSING COMPLETE!")
    
    return processor

def run_full_analysis_with_prototypes():
    """Run full analysis with integrated prototype processing."""
    
    print(f"\n" + "=" * 60)
    print("RUNNING FULL ANALYSIS WITH PROTOTYPE PROCESSING")
    print("=" * 60)
    
    # Check OpenAI availability
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        client.embeddings.create(input=["test"], model="text-embedding-3-small")
        openai_available = True
        print("✅ OpenAI API available")
    except:
        openai_available = False
        print("⚠️  OpenAI API not available, using Sentence-BERT")
    
    # Choose best available method
    if openai_available:
        embedding_model = "text-embedding-3-large"
        use_openai = True
        print(f"🚀 Using OpenAI {embedding_model} with prototype processing")
    else:
        embedding_model = "all-mpnet-base-v2"
        use_openai = False
        print(f"🚀 Using Sentence-BERT {embedding_model} with prototype processing")
    
    # Command to run
    cmd = f"""python run_modular_analysis.py \\
  --type1_dir=type1csvs \\
  --type2_csv=type2csvs \\
  --output_dir=output_integrated_prototypes \\
  --embedding_model={embedding_model} \\
  --use_prototype \\
  --use_lexical_bias_reduction \\
  --auto_k --match_only \\
  --n_clusters=5"""
    
    if use_openai:
        cmd += " \\\n  --use_openai"
    
    print(f"\n💡 COMMAND TO RUN INTEGRATED PROTOTYPE ANALYSIS:")
    print(cmd)
    
    print(f"\n📊 EXPECTED IMPROVEMENTS:")
    print(f"   • Entity replacement: Remove name/location bias")
    print(f"   • LLM variations: Reduce word choice bias") 
    print(f"   • Prototype embeddings: Focus on behavioral patterns")
    print(f"   • Better clustering: Higher silhouette scores expected")
    print(f"   • Cleaner clusters: Representative samples with [PERSON], [LOCATION]")

if __name__ == "__main__":
    # Test the system
    processor = test_integrated_prototype_system()
    
    # Show how to run full analysis
    run_full_analysis_with_prototypes()
