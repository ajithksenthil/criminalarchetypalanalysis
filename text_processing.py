import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def preprocess_text(text: str) -> str:
    """Basic text preprocessing used for life-event descriptions."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def generate_embeddings(sentences: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate sentence embeddings using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    return model.encode(sentences)
