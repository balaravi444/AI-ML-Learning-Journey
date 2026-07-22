"""
Day 63 — NLP Basics & Text Processing
Topic: NLP Real-World Applications
Date: 20 July 2026
Author: Bala Ravi

Applying NLP to real problems:
→ Text classification
→ Keyword extraction
→ Similarity matching
→ Skills extraction from job descriptions
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer)
from sklearn.metrics.pairwise import (
    cosine_similarity)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, classification_report)
import re
import warnings
warnings.filterwarnings('ignore')


def text_similarity_demo() -> None:
    """
    Cosine similarity between documents.
    Foundation of search engines + RAG!
    """
    print("=== Text Similarity (Cosine) ===\n")

    documents = [
        "Machine learning model deployment FastAPI production",
        "Deploy ML models using FastAPI and Docker",
        "Python deep learning neural network training",
        "Random forest classification scikit-learn tutorial",
        "FastAPI REST API development Python backend"
    ]

    query = "How to deploy machine learning models with FastAPI"

    all_docs = [query] + documents

    tfidf = TfidfVectorizer(
        stop_words='english')
    tfidf_matrix = tfidf.fit_transform(all_docs)

    query_vec = tfidf_matrix[0]
    doc_vecs = tfidf_matrix[1:]

    similarities = cosine_similarity(
        query_vec, doc_vecs)[0]

    print(f"Query: '{query}'\n")
    print(f"Similarity scores:")
    print(f"{'Document':<55} | {'Score':>7}")
    print("-" * 65)

    ranked = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True)

    for rank, (idx, score) in enumerate(ranked, 1):
        marker = " ← most relevant" if rank == 1 else ""
        print(f"{documents[idx][:53]:<55} | "
              f"{score:>7.4f}{marker}")

    print(f"\n💡 This is how search works!")
    print(f"   TF-IDF + cosine similarity!")
    print(f"   MemoryOS (Day 87) uses this with")
    print(f"   better embeddings (vectors)! 🔥")


def naive_bayes_text_classifier() -> None:
    """
    Naive Bayes text classifier.
    Classic NLP algorithm — fast and effective!
    """
    print("\n=== Naive Bayes Text Classifier ===\n")

    # Bug reports with priorities
    texts = [
        # Critical
        "production server down all users affected",
        "database crashed data loss occurring now",
        "security breach detected unauthorized access",
        "payment system completely broken revenue loss",
        "service outage all enterprise customers down",
        # High
        "login fails special characters password",
        "email notifications not being delivered",
        "export csv malformed output large datasets",
        "oauth google login failing thirty percent users",
        "mobile app crashes ios seventeen",
        # Medium
        "date picker wrong timezone booking form",
        "sort order resets after page refresh",
        "copy button not working firefox browser",
        "table column widths inconsistent display",
        "loading spinner stays after completion",
        # Low
        "typo error message settings page",
        "button label grammatically incorrect text",
        "footer copyright year outdated shows old",
        "minor spacing issue between form fields",
        "console warning deprecated prop component"
    ]

    labels = (
        ['Critical'] * 5 +
        ['High'] * 5 +
        ['Medium'] * 5 +
        ['Low'] * 5)

    # Vectorize
    vec = CountVectorizer(
        stop_words='english', min_df=1)
    X = vec.fit_transform(texts)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, labels, test_size=0.25,
            random_state=42,
            stratify=labels))

    # Naive Bayes
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print(f"Naive Bayes Results:")
    print(f"  Train size: {X_train.shape[0]}")
    print(f"  Test size:  {X_test.shape[0]}")
    print(f"  Accuracy:   "
          f"{accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(
        y_test, y_pred,
        zero_division=0))

    # Top words per class
    feature_names = vec.get_feature_names_out()
    print("Top words per priority class:")
    for i, cls in enumerate(nb.classes_):
        top_idx = nb.feature_log_prob_[i].argsort()[-5:]
        top_words = [feature_names[j] for j in top_idx]
        print(f"  {cls:<10}: {top_words}")

    print(f"\n💡 Naive Bayes is lightning fast!")
    print(f"   Works well with small text datasets.")
    print(f"   For larger datasets → TF-IDF + RF")
    print(f"   (what we used in Bug Predictor!)")


def skills_extractor() -> None:
    """
    Extract skills from resumes and job descriptions.
    Preview of AI Hiring Assistant (Day 67)!
    """
    print("\n=== Skills Extractor ===\n")
    print("(Preview of AI Hiring Assistant — Day 67!)\n")

    SKILL_TAXONOMY = {
        'Programming': [
            'python', 'java', 'javascript',
            'typescript', 'go', 'rust', 'c++',
            'r', 'scala', 'kotlin'],
        'ML/AI': [
            'machine learning', 'deep learning',
            'nlp', 'computer vision', 'pytorch',
            'tensorflow', 'scikit-learn', 'keras',
            'xgboost', 'hugging face', 'langchain',
            'rag', 'llm', 'transformer'],
        'Data': [
            'pandas', 'numpy', 'spark', 'hadoop',
            'sql', 'postgresql', 'mongodb',
            'elasticsearch', 'kafka', 'airflow'],
        'MLOps': [
            'mlflow', 'kubeflow', 'docker',
            'kubernetes', 'aws', 'gcp', 'azure',
            'fastapi', 'flask', 'ci/cd',
            'git', 'github'],
        'Other': [
            'communication', 'leadership',
            'problem solving', 'agile',
            'scrum', 'mentoring']
    }

    resume = """
    Experienced AI Engineer with 4 years of experience.
    Proficient in Python, PyTorch, and TensorFlow.
    Built NLP pipelines using HuggingFace Transformers.
    Experience with RAG systems and LangChain.
    Deployed ML models using FastAPI and Docker.
    Strong knowledge of SQL and PostgreSQL.
    Used MLflow for experiment tracking.
    GitHub: 50+ repositories. Active contributor.
    Good communication and problem solving skills.
    """

    jd = """
    Senior AI Engineer position requiring:
    - Python programming (5+ years)
    - Deep Learning with PyTorch or TensorFlow
    - NLP and Large Language Models experience
    - RAG and vector database knowledge
    - FastAPI for ML model deployment
    - Docker and Kubernetes for MLOps
    - SQL for data management
    - Strong problem solving skills
    """

    def extract_skills(
            text: str,
            taxonomy: dict) -> dict:
        text_lower = text.lower()
        found = {}
        for category, skills in taxonomy.items():
            matched = [
                s for s in skills
                if s in text_lower]
            if matched:
                found[category] = matched
        return found

    resume_skills = extract_skills(
        resume, SKILL_TAXONOMY)
    jd_skills = extract_skills(
        jd, SKILL_TAXONOMY)

    print("Resume Skills:")
    for cat, skills in resume_skills.items():
        print(f"  {cat}: {skills}")

    print(f"\nJob Description Required Skills:")
    for cat, skills in jd_skills.items():
        print(f"  {cat}: {skills}")

    # Match score
    all_jd = set(
        s for skills in jd_skills.values()
        for s in skills)
    all_resume = set(
        s for skills in resume_skills.values()
        for s in skills)

    matched = all_jd & all_resume
    missing = all_jd - all_resume

    match_pct = (len(matched) / len(all_jd) * 100
                  if all_jd else 0)

    print(f"\nMatch Analysis:")
    print(f"  JD requires:    {len(all_jd)} skills")
    print(f"  Resume has:     {len(all_resume)} skills")
    print(f"  Matched:        {len(matched)} skills")
    print(f"  Match score:    {match_pct:.0f}%")
    print(f"\n  ✅ Matched: {matched}")
    if missing:
        print(f"  ❌ Missing: {missing}")

    print(f"\n🚀 This IS the AI Hiring Assistant!")
    print(f"   Day 67: make this smarter with")
    print(f"   TF-IDF similarity scoring!")


if __name__ == "__main__":
    text_similarity_demo()
    naive_bayes_text_classifier()
    skills_extractor()
