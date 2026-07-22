"""
Day 63 — NLP Basics & Text Processing
Topic: Complete Text Preprocessing Pipeline
Date: 20 July 2026
Author: Bala Ravi

Every NLP project starts here!
Clean text → tokenize → normalize → features
"""
import re
import string
import warnings
warnings.filterwarnings('ignore')

try:
    import nltk
    for resource in ['punkt', 'stopwords',
                      'wordnet',
                      'averaged_perceptron_tagger',
                      'punkt_tab']:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    from nltk.tokenize import (
        word_tokenize, sent_tokenize)
    from nltk.corpus import stopwords
    from nltk.stem import (
        PorterStemmer, WordNetLemmatizer)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not installed.")
    print("    Run: pip install nltk")


def basic_clean(text: str) -> str:
    """
    Step 1: Basic text cleaning.

    Lowercase + remove noise characters.
    Keep meaningful punctuation (. ! ?)
    Remove URLs, emails, special chars.

    Args:
        text: Raw input text

    Returns:
        Cleaned text string
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+|https\S+',
        '', text)

    # Remove email addresses
    text = re.sub(
        r'\S+@\S+', '', text)

    # Remove code blocks (``` ... ```)
    text = re.sub(
        r'```[\s\S]*?```', ' code_block ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove special characters (keep . ! ? , ')
    text = re.sub(
        r'[^a-zA-Z0-9\s\.\!\?\,\']', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text: str) -> dict:
    """
    Step 2: Tokenization.

    Split text into words and sentences.
    Foundation of ALL NLP pipelines!

    Args:
        text: Cleaned text

    Returns:
        Dictionary with word and sentence tokens
    """
    if not NLTK_AVAILABLE:
        # Fallback without NLTK
        words = text.split()
        sentences = [
            s.strip() for s in
            re.split(r'[.!?]+', text)
            if s.strip()]
        return {
            'word_tokens': words,
            'sent_tokens': sentences,
            'n_words': len(words),
            'n_sentences': len(sentences)}

    word_tokens = word_tokenize(text)
    sent_tokens = sent_tokenize(text)

    return {
        'word_tokens': word_tokens,
        'sent_tokens': sent_tokens,
        'n_words': len(word_tokens),
        'n_sentences': len(sent_tokens)}


def remove_stopwords(
        tokens: list,
        extra_stopwords: list = None) -> list:
    """
    Step 3: Remove stopwords.

    Stopwords = common words with little meaning.
    "the", "is", "a", "in", "on" etc.

    Args:
        tokens: List of word tokens
        extra_stopwords: Domain-specific words to remove

    Returns:
        Filtered token list
    """
    if NLTK_AVAILABLE:
        stop_words = set(
            stopwords.words('english'))
    else:
        # Basic fallback stopword list
        stop_words = {
            'the', 'a', 'an', 'is', 'are',
            'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall',
            'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during',
            'before', 'after', 'above', 'below',
            'and', 'but', 'or', 'nor', 'so',
            'yet', 'both', 'either', 'neither',
            'not', 'only', 'own', 'same', 'than',
            'too', 'very', 'just', 'this',
            'that', 'these', 'those', 'it',
            'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'she',
            'they', 'them', 'their', 'what',
            'which', 'who', 'whom'}

    if extra_stopwords:
        stop_words.update(extra_stopwords)

    filtered = [
        token for token in tokens
        if token.lower() not in stop_words
        and token not in string.punctuation
        and len(token) > 1]

    return filtered


def stem_tokens(tokens: list) -> list:
    """
    Step 4a: Stemming.

    Chops word endings to find root.
    Fast but crude — may create non-words!

    Args:
        tokens: List of word tokens

    Returns:
        Stemmed token list
    """
    if not NLTK_AVAILABLE:
        # Very basic stemming fallback
        suffixes = ['ing', 'ed', 'es', 's', 'er']
        stemmed = []
        for token in tokens:
            t = token.lower()
            for suffix in suffixes:
                if t.endswith(suffix) and len(t) > len(suffix) + 2:
                    t = t[:-len(suffix)]
                    break
            stemmed.append(t)
        return stemmed

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens: list) -> list:
    """
    Step 4b: Lemmatization.

    Finds the actual root word (lemma).
    Slower but creates real words!

    Args:
        tokens: List of word tokens

    Returns:
        Lemmatized token list
    """
    if not NLTK_AVAILABLE:
        return tokens  # fallback: no change

    lemmatizer = WordNetLemmatizer()

    # Get POS tags for better lemmatization
    try:
        pos_tags = nltk.pos_tag(tokens)

        def get_wordnet_pos(tag: str) -> str:
            """Map POS tag to WordNet POS."""
            from nltk.corpus import wordnet
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            return wordnet.NOUN

        lemmatized = [
            lemmatizer.lemmatize(
                word,
                get_wordnet_pos(pos))
            for word, pos in pos_tags]
    except Exception:
        # Fallback: lemmatize as noun
        lemmatized = [
            lemmatizer.lemmatize(t)
            for t in tokens]

    return lemmatized


class TextPreprocessor:
    """
    Complete text preprocessing pipeline.

    Usage:
        preprocessor = TextPreprocessor()
        clean_text = preprocessor.process(raw_text)
        tokens = preprocessor.tokenize(clean_text)
    """

    def __init__(self,
                  use_lemmatization: bool = True,
                  extra_stopwords: list = None
                  ) -> None:
        """
        Initialize preprocessor.

        Args:
            use_lemmatization: True = lemmatize,
                               False = stem
            extra_stopwords: Domain-specific stops
        """
        self.use_lemmatization = use_lemmatization
        self.extra_stopwords = extra_stopwords or []

    def process(self, text: str) -> str:
        """Full preprocessing → cleaned string."""
        cleaned = basic_clean(text)
        result = tokenize_text(cleaned)
        tokens = result['word_tokens']
        tokens = remove_stopwords(
            tokens, self.extra_stopwords)

        if self.use_lemmatization:
            tokens = lemmatize_tokens(tokens)
        else:
            tokens = stem_tokens(tokens)

        return ' '.join(tokens)

    def process_batch(
            self, texts: list) -> list:
        """Process a list of texts."""
        return [self.process(t) for t in texts]


def demonstrate_pipeline() -> None:
    """Show full pipeline step by step."""
    print("=== Text Preprocessing Pipeline ===\n")

    raw_texts = [
        "Production server is DOWN!!! "
        "All users cannot login. "
        "FATAL: connection pool exhausted. "
        "Fix IMMEDIATELY!!!",

        "The login button isn't working correctly "
        "for users who have special characters "
        "in their passwords.",

        "Minor typo in the footer. "
        "Copyright year still shows 2024. "
        "Small cosmetic fix when convenient."
    ]

    preprocessor = TextPreprocessor(
        use_lemmatization=True)

    for i, text in enumerate(raw_texts, 1):
        print(f"Text {i}: {text[:60]}...")
        print(f"  Step 1 (clean):   "
              f"{basic_clean(text)[:60]}...")

        cleaned = basic_clean(text)
        tokens_result = tokenize_text(cleaned)
        tokens = tokens_result['word_tokens']
        print(f"  Step 2 (tokens):  {tokens[:8]}...")

        filtered = remove_stopwords(tokens)
        print(f"  Step 3 (no stops): {filtered[:8]}...")

        lemmatized = lemmatize_tokens(filtered)
        print(f"  Step 4 (lemma):   {lemmatized[:8]}...")

        final = preprocessor.process(text)
        print(f"  Final:            {final[:60]}...")
        print()


def stemming_vs_lemmatization() -> None:
    """Compare stemming vs lemmatization."""
    print("=== Stemming vs Lemmatization ===\n")

    test_words = [
        'running', 'studies', 'better',
        'caring', 'wolves', 'children',
        'crying', 'happily', 'databases',
        'failing', 'crashed', 'production'
    ]

    if NLTK_AVAILABLE:
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        print(f"{'Word':<15} | {'Stem':>12} | "
              f"{'Lemma (verb)':>13} | "
              f"{'Lemma (noun)':>13}")
        print("-" * 60)

        for word in test_words:
            stem = stemmer.stem(word)
            lemma_v = lemmatizer.lemmatize(
                word, pos='v')
            lemma_n = lemmatizer.lemmatize(
                word, pos='n')
            print(f"{word:<15} | {stem:>12} | "
                  f"{lemma_v:>13} | "
                  f"{lemma_n:>13}")
    else:
        print("NLTK not available — "
              "run: pip install nltk")

    print(f"\n💡 Stemming creates non-words!")
    print(f"   'studies' → 'studi' (not a word!)")
    print(f"   Lemmatization keeps real words ✅")
    print(f"\n   For production → always lemmatize!")


def job_description_preprocessor() -> None:
    """
    Real application: preprocess job descriptions.
    Preview of AI Hiring Assistant (Day 67)!
    """
    print("\n=== Job Description Preprocessor ===\n")
    print("(Preview of AI Hiring Assistant — Day 67!)\n")

    jd = """
    Senior AI Engineer — Bangalore, India

    We are looking for a passionate AI Engineer
    with 3+ years of experience in Machine Learning
    and Deep Learning. The ideal candidate will have
    strong Python programming skills and experience
    with frameworks such as TensorFlow, PyTorch,
    and Scikit-learn.

    Responsibilities:
    - Design and implement ML pipelines
    - Build and deploy AI models to production
    - Work with large datasets using Pandas and NumPy
    - Collaborate with cross-functional teams
    - Optimize model performance and scalability

    Requirements:
    - Bachelor's or Master's degree in CS or related field
    - Experience with NLP, Computer Vision, or both
    - Knowledge of MLOps tools (MLflow, Kubeflow)
    - Strong communication and problem-solving skills
    """

    # Extract skills
    skill_keywords = [
        'python', 'tensorflow', 'pytorch',
        'scikit-learn', 'pandas', 'numpy',
        'nlp', 'computer vision', 'mlops',
        'mlflow', 'kubeflow', 'machine learning',
        'deep learning', 'ai engineer']

    jd_lower = jd.lower()
    found_skills = [
        skill for skill in skill_keywords
        if skill in jd_lower]

    print(f"Job Description (excerpt):")
    print(f"{jd[:200].strip()}...")

    print(f"\nExtracted Skills ({len(found_skills)}):")
    for skill in found_skills:
        print(f"  ✅ {skill}")

    preprocessor = TextPreprocessor(
        use_lemmatization=True,
        extra_stopwords=[
            'experience', 'year', 'strong',
            'work', 'skill', 'knowledge',
            'team', 'ideal', 'candidate'])

    processed = preprocessor.process(jd)
    print(f"\nProcessed tokens (first 20):")
    print(f"  {processed.split()[:20]}")

    print(f"\n💡 This processed text becomes input")
    print(f"   to TF-IDF for resume-JD matching!")
    print(f"   That's exactly what AI Hiring")
    print(f"   Assistant (Day 67) will do! 🚀")


if __name__ == "__main__":
    demonstrate_pipeline()
    stemming_vs_lemmatization()
    job_description_preprocessor()
