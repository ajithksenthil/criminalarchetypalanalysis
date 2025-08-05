#!/usr/bin/env python3
"""
prototype_embeddings.py

Prototype-based embeddings to reduce lexical bias in event representations.
"""

import numpy as np
from typing import List, Optional
import openai

class LexicalBiasFixer:
    """Fix lexical bias in embeddings using prototype-based approach."""
    
    def __init__(self, client=None):
        self.client = client
    
    def generate_lexical_variations(self, text: str, num_variants: int = 5) -> List[str]:
        """
        Generate lexical variations of text to capture semantic meaning.
        
        Args:
            text: Original event text
            num_variants: Number of variations to generate
            
        Returns:
            List of lexical variations
        """
        if not self.client:
            print("[INFO] No LLM client available, using original text only")
            return [text]  # Fallback to original text
        
        prompt = (
            f"Generate {num_variants} alternative versions of the following sentence, "
            f"using synonyms and varied phrasing, while preserving the meaning:\n\n"
            f"{text}\n\nAlternative versions:"
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            reply = response.choices[0].message.content.strip()
            variations = [line.strip() for line in reply.split("\n") if line.strip()]
            
            if not variations:
                return [text]
            
            return variations
        except Exception as e:
            print(f"[WARNING] Could not generate lexical variations: {e}")
            return [text]
    
    def get_prototype_embedding(self, event_text: str, model, num_variants: int = 5) -> np.ndarray:
        """
        Get prototype embedding by averaging embeddings of lexical variations.
        This reduces bias from specific word choices.
        
        Args:
            event_text: Original event text
            model: Sentence embedding model
            num_variants: Number of lexical variations to generate
            
        Returns:
            Prototype embedding (centroid of variations)
        """
        # Generate lexical variations
        variations = self.generate_lexical_variations(event_text, num_variants)
        
        # Include original text
        all_versions = variations + [event_text]
        
        # Compute embeddings for all versions
        embeddings = model.encode(all_versions)
        
        # Return centroid (prototype) embedding
        return np.mean(embeddings, axis=0)

class PrototypeEmbeddingProcessor:
    """Process events using prototype embeddings to reduce lexical bias."""
    
    def __init__(self, use_prototype: bool = True, client=None):
        self.use_prototype = use_prototype
        self.bias_fixer = LexicalBiasFixer(client) if use_prototype else None
    
    def process_events(self, events: List[str], model, num_variants: int = 3) -> np.ndarray:
        """
        Process events using prototype embeddings if enabled, otherwise standard embeddings.
        
        Args:
            events: List of event texts
            model: Sentence embedding model
            num_variants: Number of lexical variations per event
            
        Returns:
            Event embeddings matrix
        """
        if not self.use_prototype or not self.bias_fixer:
            print("[INFO] Using standard embeddings (no prototype processing)")
            return model.encode(events)
        
        print(f"[INFO] Processing {len(events)} events with prototype embeddings...")
        
        prototype_embeddings = []
        for i, event_text in enumerate(events):
            if i % 100 == 0:
                print(f"  Processing event {i+1}/{len(events)}")
            
            # Get prototype embedding for this event
            prototype_emb = self.bias_fixer.get_prototype_embedding(event_text, model, num_variants)
            prototype_embeddings.append(prototype_emb)
        
        print("[INFO] Prototype embedding processing complete")
        return np.array(prototype_embeddings)

def create_prototype_processor(use_prototype: bool = False, 
                             openai_api_key: Optional[str] = None) -> PrototypeEmbeddingProcessor:
    """
    Create a prototype embedding processor.
    
    Args:
        use_prototype: Whether to use prototype embeddings
        openai_api_key: OpenAI API key for lexical variation generation
        
    Returns:
        PrototypeEmbeddingProcessor instance
    """
    client = None
    if use_prototype and openai_api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_api_key)
            print("[INFO] OpenAI client initialized for lexical variation generation")
        except Exception as e:
            print(f"[WARNING] Could not initialize OpenAI client: {e}")
            print("[INFO] Falling back to standard embeddings")
            use_prototype = False
    elif use_prototype:
        print("[WARNING] Prototype embeddings requested but no OpenAI API key provided")
        print("[INFO] Falling back to standard embeddings")
        use_prototype = False
    
    return PrototypeEmbeddingProcessor(use_prototype, client)

# Backward compatibility functions
def get_imputed_embedding(event_text: str, model, num_variants: int = 5, client=None) -> np.ndarray:
    """
    Backward compatibility function for get_imputed_embedding.
    """
    bias_fixer = LexicalBiasFixer(client)
    return bias_fixer.get_prototype_embedding(event_text, model, num_variants)

def generate_lexical_variations(text: str, num_variants: int = 5, client=None) -> List[str]:
    """
    Backward compatibility function for generate_lexical_variations.
    """
    bias_fixer = LexicalBiasFixer(client)
    return bias_fixer.generate_lexical_variations(text, num_variants)
