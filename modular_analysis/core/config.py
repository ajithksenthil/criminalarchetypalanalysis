#!/usr/bin/env python3
"""
config.py

Configuration and global settings for the criminal archetypal analysis system.
"""

import os
import random
import numpy as np
try:
    import openai
except ImportError:
    openai = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None

# Global settings
RANDOM_SEED = 42
DEFAULT_N_CLUSTERS = 5
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_MAX_TFIDF_FEATURES = 500

# File patterns
TYPE1_FILE_PATTERN = "Type1_*.csv"
TYPE2_FILE_PATTERN = "Type2_*.csv"

# Analysis thresholds
MIN_CRIMINALS_FOR_ANALYSIS = 5
DEFAULT_DIFF_THRESHOLD = 0.1
DEFAULT_MIN_EFFECT_SIZE = 0.1

# Clustering parameters
DEFAULT_K_RANGE = (3, 20)
DEFAULT_KMEANS_INIT = 10

def ensure_nltk_data():
    """Download required NLTK data if not present."""
    if nltk is None:
        print("Warning: NLTK not available")
        return

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab/english/",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except:
                print(f"Warning: Could not download NLTK resource {pkg}")

def setup_environment():
    """Set up the global environment for reproducible analysis."""
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Set up OpenAI client if API key is available
    client = None
    if openai and "OPENAI_API_KEY" in os.environ:
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            client = openai.OpenAI()
        except Exception as e:
            print(f"Warning: Could not set up OpenAI client: {e}")

    return client

def get_clustering_config():
    """Get clustering configuration from environment variables."""
    return {
        'use_improved': os.environ.get("USE_IMPROVED_CLUSTERING", "0") == "1",
        'method': os.environ.get("CLUSTERING_METHOD", "kmeans"),
        'auto_select_k': os.environ.get("AUTO_SELECT_K", "0") == "1",
        'reduce_dims': os.environ.get("REDUCE_DIMENSIONS", "1") == "1"
    }

class AnalysisConfig:
    """Configuration class for analysis parameters."""
    
    def __init__(self, **kwargs):
        # Data paths
        self.type1_dir = kwargs.get('type1_dir')
        self.type2_csv = kwargs.get('type2_csv')
        self.output_dir = kwargs.get('output_dir', 'output')
        
        # Clustering parameters
        self.n_clusters = kwargs.get('n_clusters', DEFAULT_N_CLUSTERS)
        self.auto_k = kwargs.get('auto_k', False)
        
        # Analysis options
        self.no_llm = kwargs.get('no_llm', False)
        self.multi_modal = kwargs.get('multi_modal', False)
        self.train_proto_net = kwargs.get('train_proto_net', False)
        self.use_tfidf = kwargs.get('use_tfidf', False)
        self.use_prototype = kwargs.get('use_prototype', False)
        self.openai_api_key = kwargs.get('openai_api_key', None)
        self.use_statistical_validation = kwargs.get('use_statistical_validation', False)
        self.embedding_model = kwargs.get('embedding_model', 'all-MiniLM-L6-v2')
        self.use_openai = kwargs.get('use_openai', False)
        self.openai_model = kwargs.get('openai_model', 'text-embedding-3-large')
        self.use_lexical_bias_reduction = kwargs.get('use_lexical_bias_reduction', False)
        self.match_only = kwargs.get('match_only', False)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'type1_dir': self.type1_dir,
            'type2_csv': self.type2_csv,
            'output_dir': self.output_dir,
            'n_clusters': self.n_clusters,
            'auto_k': self.auto_k,
            'no_llm': self.no_llm,
            'multi_modal': self.multi_modal,
            'train_proto_net': self.train_proto_net,
            'use_tfidf': self.use_tfidf,
            'use_prototype': self.use_prototype,
            'openai_api_key': self.openai_api_key,
            'use_statistical_validation': self.use_statistical_validation,
            'embedding_model': self.embedding_model,
            'use_openai': self.use_openai,
            'openai_model': self.openai_model,
            'use_lexical_bias_reduction': self.use_lexical_bias_reduction,
            'match_only': self.match_only
        }
    
    @classmethod
    def from_args(cls, args):
        """Create configuration from command line arguments."""
        return cls(
            type1_dir=args.type1_dir,
            type2_csv=args.type2_csv,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
            auto_k=args.auto_k,
            no_llm=args.no_llm,
            multi_modal=args.multi_modal,
            train_proto_net=args.train_proto_net,
            use_tfidf=args.use_tfidf,
            use_prototype=args.use_prototype,
            openai_api_key=args.openai_api_key,
            use_statistical_validation=args.use_statistical_validation,
            embedding_model=args.embedding_model,
            use_openai=args.use_openai,
            openai_model=args.openai_model,
            use_lexical_bias_reduction=args.use_lexical_bias_reduction,
            match_only=args.match_only
        )
