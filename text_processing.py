import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import openai
    if "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    _openai_client = openai.OpenAI()
except Exception:
    openai = None  # type: ignore
    _openai_client = None

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


def generate_lexical_variations(text: str, num_variants: int = 5) -> list:
    """Return alternative phrasings of `text` using an LLM if available."""
    if _openai_client is None:
        return [text]
    prompt = (
        "Generate {} alternative versions of the following sentence, "
        "using synonyms and varied phrasing, while preserving the meaning:\n\n".format(num_variants)
        + text
        + "\n\nAlternative versions:"
    )
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content.strip()
        variations = [ln.strip() for ln in reply.split("\n") if ln.strip()]
        return variations or [text]
    except Exception as e:
        print(f"[WARNING] Failed to generate lexical variations: {e}")
        return [text]


def get_imputed_embedding(event_text: str, model, num_variants: int = 5) -> np.ndarray:
    """Compute an averaged embedding over lexical variations of `event_text`."""
    variants = generate_lexical_variations(event_text, num_variants=num_variants)
    all_texts = variants + [event_text]
    embeddings = model.encode(all_texts)
    return np.mean(embeddings, axis=0)


def generate_imputed_embeddings(sentences: list, model_name: str = "all-MiniLM-L6-v2", num_variants: int = 5) -> np.ndarray:
    """Generate embeddings using lexical imputation for each sentence."""
    model = SentenceTransformer(model_name)
    embs = [get_imputed_embedding(s, model, num_variants=num_variants) for s in sentences]
    return np.vstack(embs)
