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

try:
    from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_MAX_TFIDF_FEATURES
except ImportError:
    try:
        from ..core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_MAX_TFIDF_FEATURES
    except ImportError:
        DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
        DEFAULT_MAX_TFIDF_FEATURES = 500

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
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, use_tfidf: bool = False):
        self.model_name = model_name
        self.use_tfidf = use_tfidf
        
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_TFIDF_FEATURES)
            self.model = None
        else:
            self.model = SentenceTransformer(model_name)
            self.vectorizer = None
    
    def generate_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            Embedding matrix
        """
        if self.use_tfidf:
            return self._generate_tfidf_embeddings(sentences)
        else:
            return self._generate_transformer_embeddings(sentences)
    
    def _generate_tfidf_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings."""
        vectors = self.vectorizer.fit_transform(sentences)
        return vectors.toarray()
    
    def _generate_transformer_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generate transformer-based embeddings."""
        return self.model.encode(sentences)

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
                 use_augmentation: bool = False, client=None):
        super().__init__(model_name, use_tfidf)
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
        elif self.use_augmentation and self.augmenter:
            return self._generate_augmented_embeddings(sentences)
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
