#!/usr/bin/env python3
"""
text_processing.py

Text preprocessing and embedding functionality.
"""

import re
import numpy as np
from typing import List, Optional
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants to avoid import issues
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_MAX_TFIDF_FEATURES = 500

# State-of-the-art embedding models
RECOMMENDED_MODELS = {
    "all-MiniLM-L6-v2": "Current baseline (384 dim)",
    "all-mpnet-base-v2": "Best general performance (768 dim)",
    "all-MiniLM-L12-v2": "Improved version (384 dim)",
    "paraphrase-mpnet-base-v2": "Good for similar events (768 dim)",
    "all-distilroberta-v1": "Fast and good (768 dim)",
    "multi-qa-mpnet-base-dot-v1": "Good for diverse text (768 dim)"
}

class TextPreprocessor:
    """Text preprocessing functionality."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercasing, remove digits/punctuation, tokenize,
        remove stopwords, and lemmatize.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove digits and punctuation
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        
        return " ".join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]

class EmbeddingGenerator:
    """Generate embeddings for text data."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, use_tfidf: bool = False,
                 use_prototype: bool = False, openai_api_key: Optional[str] = None,
                 use_openai: bool = False, use_lexical_bias_reduction: bool = False):
        self.model_name = model_name
        self.use_tfidf = use_tfidf
        self.use_prototype = use_prototype
        self.use_openai = use_openai
        self.use_lexical_bias_reduction = use_lexical_bias_reduction

        if use_tfidf:
            self.vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_TFIDF_FEATURES)
            self.model = None
            self.openai_processor = None
        elif use_openai:
            # Use OpenAI embeddings
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from openai_integration import create_openai_embedding_generator

                self.openai_processor = create_openai_embedding_generator(
                    use_prototype=use_prototype,
                    embedding_model=model_name if model_name.startswith('text-embedding-') else 'text-embedding-3-large'
                )
                self.model = None
                self.vectorizer = None
                print(f"[INFO] Using OpenAI embeddings: {model_name}")

            except ImportError as e:
                print(f"[WARNING] Could not import OpenAI integration: {e}")
                print("[INFO] Falling back to Sentence Transformers")
                self.use_openai = False
                self.model = SentenceTransformer(model_name)
                self.vectorizer = None
                self.openai_processor = None
        else:
            self.model = SentenceTransformer(model_name)
            self.vectorizer = None
            self.openai_processor = None

        # Initialize prototype processor if requested
        self.prototype_processor = None
        if use_prototype and not use_tfidf:
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from embeddings.prototype_embeddings import create_prototype_processor
                self.prototype_processor = create_prototype_processor(use_prototype, openai_api_key)
                print("[INFO] Prototype embedding processor initialized")
            except ImportError as e:
                print(f"[WARNING] Could not import prototype embeddings: {e}")
                print("[INFO] Using standard embeddings")

        # Initialize lexical bias processor if requested
        self.lexical_bias_processor = None
        if use_lexical_bias_reduction:
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                from lexical_bias_processor import LexicalBiasProcessor

                self.lexical_bias_processor = LexicalBiasProcessor()
                print("[INFO] Lexical bias processor initialized")

            except ImportError as e:
                print(f"[WARNING] Could not import lexical bias processor: {e}")
                print("[INFO] Proceeding without lexical bias reduction")
                self.use_prototype = False
    
    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.

        Args:
            sentences: List of input sentences

        Returns:
            Embedding matrix
        """
        # Apply lexical bias reduction if requested
        self.processed_sentences = sentences  # Store original by default
        if self.use_lexical_bias_reduction and self.lexical_bias_processor:
            print("[INFO] Applying lexical bias reduction...")
            self.processed_sentences = self.lexical_bias_processor.process_event_list(sentences)

        if self.use_tfidf:
            return self._generate_tfidf_embeddings(self.processed_sentences)
        elif self.use_openai and self.openai_processor:
            return self._generate_openai_embeddings(self.processed_sentences)
        elif self.use_prototype and self.prototype_processor:
            return self._generate_prototype_embeddings(self.processed_sentences)
        else:
            return self._generate_transformer_embeddings(self.processed_sentences)

    def get_processed_sentences(self) -> List[str]:
        """Get the processed sentences (with bias reduction if applied)."""
        return getattr(self, 'processed_sentences', [])
    
    def _generate_tfidf_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings."""
        vectors = self.vectorizer.fit_transform(sentences)
        return vectors.toarray()
    
    def _generate_transformer_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate transformer-based embeddings."""
        return self.model.encode(sentences)

    def _generate_prototype_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate prototype embeddings to reduce lexical bias."""
        return self.prototype_processor.process_events(sentences, self.model)

    def _generate_openai_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate OpenAI embeddings."""
        return self.openai_processor.generate_embeddings(sentences)

class LexicalAugmenter:
    """Lexical augmentation using LLM for improved embeddings."""
    
    def __init__(self, client=None):
        self.client = client
    
    def generate_lexical_variations(self, text: str, num_variants: int = 5) -> List[str]:
        """
        Generate lexical variations of text using LLM.
        
        Args:
            text: Input text
            num_variants: Number of variations to generate
            
        Returns:
            List of text variations
        """
        if not self.client:
            return [text]  # Return original if no LLM available
        
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
            
            return variations if variations else [text]
            
        except Exception as e:
            print(f"[ERROR] Generating lexical variations: {e}")
            return [text]
    
    def get_imputed_embedding(self, event_text: str, model: SentenceTransformer, num_variants: int = 5) -> np.ndarray:
        """
        Generate an imputed embedding using lexical variations.
        
        Args:
            event_text: Input event text
            model: SentenceTransformer model
            num_variants: Number of variations to generate
            
        Returns:
            Average embedding across variations
        """
        # Generate variations
        variants = self.generate_lexical_variations(event_text, num_variants)
        
        # Include original text
        all_versions = variants + [event_text]
        
        # Compute embeddings for all versions
        embeddings = model.encode(all_versions)
        
        # Return average embedding
        return np.mean(embeddings, axis=0)

class AdvancedEmbeddingGenerator(EmbeddingGenerator):
    """Advanced embedding generator with lexical augmentation."""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, use_tfidf: bool = False,
                 use_augmentation: bool = False, client=None, use_openai: bool = False,
                 use_prototype: bool = False, use_lexical_bias_reduction: bool = False):
        super().__init__(model_name, use_tfidf, use_prototype=use_prototype, use_openai=use_openai,
                        use_lexical_bias_reduction=use_lexical_bias_reduction)
        self.use_augmentation = use_augmentation
        self.augmenter = LexicalAugmenter(client) if use_augmentation else None
    
    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings with optional lexical augmentation.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            Embedding matrix
        """
        if self.use_tfidf:
            return self._generate_tfidf_embeddings(sentences)
        elif self.use_openai and self.openai_processor:
            return self._generate_openai_embeddings(sentences)
        elif self.use_augmentation and self.augmenter:
            return self._generate_augmented_embeddings(sentences)
        elif self.use_prototype and self.prototype_processor:
            return self._generate_prototype_embeddings(sentences)
        else:
            return self._generate_transformer_embeddings(sentences)
    
    def _generate_augmented_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings with lexical augmentation."""
        embeddings = []
        total = len(sentences)
        
        print(f"[INFO] Generating augmented embeddings for {total} sentences...")
        
        for idx, sentence in enumerate(sentences):
            if idx % 50 == 0:
                print(f"[INFO] Processing sentence {idx+1}/{total}")
            
            avg_embedding = self.augmenter.get_imputed_embedding(sentence, self.model)
            embeddings.append(avg_embedding)
        
        return np.array(embeddings)

class TextAnalyzer:
    """Analyze text data for insights."""
    
    @staticmethod
    def find_representative_samples(sentences: List[str], embeddings: np.ndarray, 
                                  labels: List[int], n_reps: int = 3) -> List[dict]:
        """
        Find representative samples for each cluster.
        
        Args:
            sentences: Original sentences
            embeddings: Sentence embeddings
            labels: Cluster labels
            n_reps: Number of representatives per cluster
            
        Returns:
            List of cluster information dictionaries
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        clusters_dict = {}
        for idx, label in enumerate(labels):
            clusters_dict.setdefault(label, []).append(idx)
        
        results = []
        for label, indices in clusters_dict.items():
            cluster_embs = embeddings[indices]
            centroid = np.mean(cluster_embs, axis=0)
            sims = cosine_similarity([centroid], cluster_embs)[0]
            sorted_idx = np.argsort(sims)[::-1]
            rep_indices = [indices[i] for i in sorted_idx[:n_reps]]
            rep_texts = [sentences[i] for i in rep_indices]
            
            results.append({
                "cluster_id": label,
                "size": len(indices),
                "representative_samples": rep_texts
            })
        
        results.sort(key=lambda x: x["cluster_id"])
        return results

# Backward compatibility aliases
def preprocess_text(text: str) -> str:
    """Backward compatibility alias for TextPreprocessor.preprocess_text()."""
    processor = TextPreprocessor()
    return processor.preprocess_text(text)

def generate_embeddings(sentences: List[str], use_tfidf: bool = False) -> np.ndarray:
    """Backward compatibility alias for EmbeddingGenerator.generate_embeddings()."""
    generator = EmbeddingGenerator(use_tfidf=use_tfidf)
    return generator.generate_embeddings(sentences)

def generate_tfidf_embeddings(sentences: List[str]) -> np.ndarray:
    """Generate TF-IDF embeddings."""
    return generate_embeddings(sentences, use_tfidf=True)

def get_imputed_embedding(event_text: str, model, num_variants: int = 5) -> np.ndarray:
    """Backward compatibility alias for lexical augmentation."""
    augmenter = LexicalAugmenter()
    return augmenter.get_imputed_embedding(event_text, model, num_variants)

def find_representative_samples(sentences: List[str], embeddings: np.ndarray,
                              labels: List[int], n_reps: int = 3) -> List[dict]:
    """Backward compatibility alias."""
    return TextAnalyzer.find_representative_samples(sentences, embeddings, labels, n_reps)
