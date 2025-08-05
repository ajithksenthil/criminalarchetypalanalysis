#!/usr/bin/env python3
"""
openai_integration.py

Comprehensive OpenAI integration for:
1. State-of-the-art text embeddings (text-embedding-3-large)
2. Prototype embedding calculations with lexical variations
3. Intelligent cluster labeling with GPT-4
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

class OpenAIEmbeddingProcessor:
    """High-quality OpenAI embeddings for criminal life events."""
    
    def __init__(self, model: str = "text-embedding-3-large"):
        """
        Initialize OpenAI embedding processor.
        
        Args:
            model: OpenAI embedding model to use
                  - text-embedding-3-large: 3072 dim, highest quality
                  - text-embedding-3-small: 1536 dim, good balance
                  - text-embedding-ada-002: 1536 dim, legacy
        """
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        print(f"[INFO] Initialized OpenAI embeddings with {model}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generate high-quality OpenAI embeddings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for API calls (max 2048 for OpenAI)
            
        Returns:
            Embedding matrix (n_texts, embedding_dim)
        """
        print(f"[INFO] Generating OpenAI embeddings for {len(texts)} texts using {self.model}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting - OpenAI allows 3000 RPM for tier 1
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  ❌ Error in batch {batch_num}: {e}")
                # Fallback: create zero embeddings for this batch
                embedding_dim = 3072 if "large" in self.model else 1536
                batch_embeddings = [np.zeros(embedding_dim).tolist() for _ in batch]
                all_embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(all_embeddings)
        print(f"[INFO] Generated embeddings shape: {embeddings_array.shape}")
        
        return embeddings_array

class OpenAIPrototypeProcessor:
    """Generate prototype embeddings using OpenAI for lexical variation and embedding."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.embedding_processor = OpenAIEmbeddingProcessor(embedding_model)
        
        print(f"[INFO] Initialized OpenAI prototype processor")
    
    def generate_lexical_variations(self, text: str, num_variants: int = 5) -> List[str]:
        """
        Generate lexical variations using GPT-4 to reduce word choice bias.
        
        Args:
            text: Original life event description
            num_variants: Number of variations to generate
            
        Returns:
            List of lexical variations
        """
        prompt = f"""Generate {num_variants} alternative versions of the following criminal life event description. 
Use different words and phrasing while preserving the exact meaning and factual content.
Focus on varying vocabulary, sentence structure, and expression style.

Original: {text}

Alternative versions:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective for this task
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Parse variations (assume each line is a variation)
            variations = [line.strip() for line in reply.split('\n') if line.strip()]
            
            # Clean up numbered lists
            cleaned_variations = []
            for var in variations:
                # Remove leading numbers/bullets
                cleaned = var.lstrip('0123456789.-) ').strip()
                if cleaned and len(cleaned) > 10:  # Reasonable length check
                    cleaned_variations.append(cleaned)
            
            return cleaned_variations[:num_variants]
            
        except Exception as e:
            print(f"[WARNING] Could not generate lexical variations: {e}")
            return [text]  # Fallback to original
    
    def get_prototype_embedding(self, event_text: str, num_variants: int = 3) -> np.ndarray:
        """
        Generate prototype embedding by averaging embeddings of lexical variations.
        
        Args:
            event_text: Original event text
            num_variants: Number of lexical variations to generate
            
        Returns:
            Prototype embedding (centroid of variations)
        """
        # Generate lexical variations
        variations = self.generate_lexical_variations(event_text, num_variants)
        
        # Include original text
        all_versions = [event_text] + variations
        
        # Generate embeddings for all versions
        embeddings = self.embedding_processor.generate_embeddings(all_versions)
        
        # Return centroid (prototype) embedding
        prototype = np.mean(embeddings, axis=0)
        
        return prototype
    
    def process_events_with_prototypes(self, events: List[str],
                                     num_variants: int = 3) -> np.ndarray:
        """
        Process all events using prototype embeddings with entity replacement.

        Args:
            events: List of life event descriptions
            num_variants: Number of lexical variations per event

        Returns:
            Prototype embeddings matrix
        """
        print(f"[INFO] Processing {len(events)} events with integrated prototype embeddings")
        print(f"[INFO] Step 1: Entity replacement (names, locations → generic labels)")
        print(f"[INFO] Step 2: Generate {num_variants} lexical variations per event")
        print(f"[INFO] Step 3: Create prototype embeddings (average variations)")

        # Initialize lexical bias processor
        try:
            from lexical_bias_processor import LexicalBiasProcessor
            bias_processor = LexicalBiasProcessor()
            print(f"[INFO] Lexical bias processor initialized")
        except ImportError:
            print(f"[WARNING] Lexical bias processor not available")
            bias_processor = None

        prototype_embeddings = []

        for i, event_text in enumerate(events):
            if i % 50 == 0:
                print(f"  Processing event {i+1}/{len(events)}")

            # Step 1: Apply entity replacement if available
            processed_text = event_text
            if bias_processor:
                processed_text = bias_processor.process_text(event_text)

            # Step 2 & 3: Generate prototype embedding from processed text
            prototype_emb = self.get_prototype_embedding(processed_text, num_variants)
            prototype_embeddings.append(prototype_emb)

        print(f"[INFO] Integrated prototype embedding processing complete")
        return np.array(prototype_embeddings)

class OpenAIClusterLabeler:
    """Intelligent cluster labeling using GPT-4."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        print(f"[INFO] Initialized OpenAI cluster labeler")
    
    def label_cluster(self, representative_samples: List[str], 
                     cluster_id: int, cluster_size: int) -> Dict[str, str]:
        """
        Generate intelligent cluster label using GPT-4.
        
        Args:
            representative_samples: Representative events from the cluster
            cluster_id: Cluster identifier
            cluster_size: Number of events in cluster
            
        Returns:
            Dictionary with cluster theme and detailed description
        """
        # Prepare samples for analysis
        samples_text = "\n".join([f"{i+1}. {sample}" for i, sample in enumerate(representative_samples)])
        
        prompt = f"""Analyze these {len(representative_samples)} representative criminal life events from Cluster {cluster_id} (containing {cluster_size} total events).

Representative Events:
{samples_text}

Based on these examples, provide:
1. A concise archetypal theme (2-4 words, e.g., "Violent Crimes", "Early Life Events", "Legal Proceedings")
2. A detailed description of what this cluster represents in criminal behavioral patterns

Respond in JSON format:
{{
    "archetypal_theme": "Brief theme name",
    "detailed_description": "Comprehensive description of the behavioral pattern this cluster represents",
    "key_characteristics": ["characteristic 1", "characteristic 2", "characteristic 3"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4 for best analysis
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(reply)
                return result
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                return {
                    "archetypal_theme": "Mixed Behavioral Events",
                    "detailed_description": reply,
                    "key_characteristics": ["Various criminal behaviors"]
                }
                
        except Exception as e:
            print(f"[WARNING] Could not label cluster {cluster_id}: {e}")
            return {
                "archetypal_theme": f"Cluster {cluster_id} Events",
                "detailed_description": "Unable to generate automatic description",
                "key_characteristics": ["Requires manual analysis"]
            }
    
    def label_all_clusters(self, cluster_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Label all clusters with intelligent themes.
        
        Args:
            cluster_info: List of cluster information dictionaries
            
        Returns:
            Updated cluster information with OpenAI labels
        """
        print(f"[INFO] Labeling {len(cluster_info)} clusters with OpenAI")
        
        labeled_clusters = []
        
        for cluster in cluster_info:
            cluster_id = cluster['cluster_id']
            cluster_size = cluster['size']
            samples = cluster['representative_samples']
            
            print(f"  Labeling Cluster {cluster_id} ({cluster_size} events)...")
            
            # Get OpenAI label
            label_info = self.label_cluster(samples, cluster_id, cluster_size)
            
            # Update cluster info
            updated_cluster = cluster.copy()
            updated_cluster.update({
                'archetypal_theme': label_info['archetypal_theme'],
                'detailed_description': label_info['detailed_description'],
                'key_characteristics': label_info['key_characteristics'],
                'labeling_method': 'OpenAI GPT-4'
            })
            
            labeled_clusters.append(updated_cluster)
            
            print(f"    → {label_info['archetypal_theme']}")
        
        print(f"[INFO] Cluster labeling complete")
        return labeled_clusters

def create_openai_embedding_generator(use_prototype: bool = False, 
                                    embedding_model: str = "text-embedding-3-large") -> Any:
    """
    Create OpenAI-based embedding generator for the modular system.
    
    Args:
        use_prototype: Whether to use prototype embeddings
        embedding_model: OpenAI embedding model to use
        
    Returns:
        Embedding generator compatible with the modular system
    """
    
    class OpenAIEmbeddingGenerator:
        """OpenAI embedding generator compatible with modular system."""
        
        def __init__(self):
            self.use_prototype = use_prototype
            self.embedding_model = embedding_model
            
            if use_prototype:
                self.processor = OpenAIPrototypeProcessor(embedding_model)
            else:
                self.processor = OpenAIEmbeddingProcessor(embedding_model)
        
        def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
            """Generate embeddings compatible with modular system."""
            if self.use_prototype:
                return self.processor.process_events_with_prototypes(sentences)
            else:
                return self.processor.generate_embeddings(sentences)
    
    return OpenAIEmbeddingGenerator()

def test_openai_integration():
    """Test OpenAI integration with sample data."""
    print("Testing OpenAI Integration...")
    
    # Test data
    sample_events = [
        "Born in Chicago to working-class parents",
        "First arrest for burglary at age 16", 
        "Served 3 years in state prison",
        "Murdered victim in downtown area",
        "Trial resulted in life sentence"
    ]
    
    # Test embeddings
    print("\n1. Testing OpenAI Embeddings...")
    embedding_proc = OpenAIEmbeddingProcessor()
    embeddings = embedding_proc.generate_embeddings(sample_events)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test prototype embeddings
    print("\n2. Testing Prototype Embeddings...")
    prototype_proc = OpenAIPrototypeProcessor()
    prototype_emb = prototype_proc.get_prototype_embedding(sample_events[0])
    print(f"Prototype embedding shape: {prototype_emb.shape}")
    
    # Test cluster labeling
    print("\n3. Testing Cluster Labeling...")
    labeler = OpenAIClusterLabeler()
    label_result = labeler.label_cluster(sample_events[:3], 0, 100)
    print(f"Cluster label: {label_result}")
    
    print("\n✅ OpenAI integration test complete!")

if __name__ == "__main__":
    test_openai_integration()
