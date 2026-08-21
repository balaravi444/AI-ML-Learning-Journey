"""
Day 82 — HuggingFace: Pretrained Models
Topic: HuggingFace Pipelines
Date: 08 August 2026
Author: Bala Ravi

500K+ models, 3 lines of code.
State-of-the-art NLP without PhD!
"""
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  transformers not installed.")
    print("    Run: pip install transformers\n")


def sentiment_analysis_demo() -> None:
    """HuggingFace sentiment analysis."""
    print("=== Sentiment Analysis Pipeline ===\n")

    if not HF_AVAILABLE:
        print("transformers not available.")
        print("\nExpected output:\n")
        examples = [
            ("This AI project is absolutely amazing!",
             "POSITIVE", 0.9998),
            ("The server crashed and all data lost.",
             "NEGATIVE", 0.9987),
            ("The meeting is scheduled for Monday.",
             "NEUTRAL", 0.9843),
            ("I love building real products!",
             "POSITIVE", 0.9995)
        ]
        for text, label, score in examples:
            emoji = "✅" if label == "POSITIVE" else (
                "❌" if label == "NEGATIVE" else "➡️")
            print(f"  {emoji} {text[:45]}")
            print(f"     → {label} ({score:.4f})\n")
        return

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english")

    texts = [
        "This AI project is absolutely amazing!",
        "The server crashed and all data lost.",
        "The meeting is scheduled for Monday.",
        "I love building real products from scratch!",
        "Production is completely down. This is terrible.",
        "Minor typo in the footer, easy fix."
    ]

    print(f"{'Text':<50} | {'Label':>9} | {'Score':>7}")
    print("-" * 70)

    for text in texts:
        result = classifier(text)[0]
        emoji = ("✅" if result['label'] == 'POSITIVE'
                  else "❌")
        print(f"{text[:48]:<50} | "
              f"{result['label']:>9} | "
              f"{result['score']:>7.4f} {emoji}")


def ner_demo() -> None:
    """Named Entity Recognition with HuggingFace."""
    print("\n=== NER Pipeline ===\n")

    if not HF_AVAILABLE:
        print("transformers not available.")
        print("\nExpected output:\n")

        ner_results = [
            ("Bala Ravi studies at Oxford College "
             "in Bangalore, India",
             [("PER", "Bala Ravi"),
              ("ORG", "Oxford College"),
              ("LOC", "Bangalore"),
              ("LOC", "India")]),
            ("The auth-service on AWS crashed at 3am UTC",
             [("ORG", "AWS"),
              ("MISC", "UTC")])
        ]
        for text, entities in ner_results:
            print(f"Text: {text}")
            for ent_type, ent_text in entities:
                print(f"  [{ent_type}] {ent_text}")
            print()
        return

    ner = pipeline(
        "ner",
        aggregation_strategy="simple")

    texts = [
        "Bala Ravi studies at The Oxford College "
        "of Science in Bangalore, India.",
        "Microsoft Azure reported an outage in "
        "East US affecting Google and Amazon services.",
        "The auth-service v2.3.1 crashed on "
        "AWS us-east-1 at 3am UTC today."
    ]

    for text in texts:
        print(f"Text: {text}")
        entities = ner(text)
        for ent in entities:
            print(f"  [{ent['entity_group']:>4}] "
                  f"'{ent['word']}' "
                  f"({ent['score']:.3f})")
        print()


def question_answering_demo() -> None:
    """Question Answering pipeline."""
    print("\n=== Question Answering Pipeline ===\n")

    if not HF_AVAILABLE:
        print("transformers not available.")
        print("\nExpected QA output:\n")

        qa_examples = [
            ("What is melanoma?",
             "the deadliest form of skin cancer", 0.94),
            ("What is the survival rate if caught early?",
             "99%", 0.97),
            ("Who built the skin disease detector?",
             "Bala Ravi", 0.93)
        ]
        for q, a, score in qa_examples:
            print(f"  Q: {q}")
            print(f"  A: {a} ({score:.2f})\n")
        return

    qa = pipeline("question-answering")

    context = """
    Melanoma is the deadliest form of skin cancer.
    It develops from melanocytes, the cells that
    give skin its color. If caught early, the
    survival rate is 99%. If caught late, it drops
    to just 15%. Bala Ravi built a skin disease
    detector using MobileNetV2 and transfer learning
    as part of his 90-day AI/ML learning journey.
    The model achieves 89% accuracy on 7 disease
    classes including melanoma detection.
    """

    questions = [
        "What is melanoma?",
        "What is the survival rate if caught early?",
        "Who built the skin disease detector?",
        "What accuracy does the model achieve?"
    ]

    for q in questions:
        result = qa(question=q, context=context)
        print(f"Q: {q}")
        print(f"A: {result['answer']} "
              f"(confidence: {result['score']:.3f})")
        print()


def zero_shot_classification() -> None:
    """Zero-shot: classify without training data!"""
    print("\n=== Zero-Shot Classification ===\n")
    print("Classify into ANY categories without training!\n")

    if not HF_AVAILABLE:
        print("transformers not available.")
        print("\nExpected zero-shot results:\n")

        examples = [
            ("Production database down, all users affected",
             "critical", 0.89),
            ("Minor typo in footer copyright year",
             "low priority", 0.95),
            ("Login broken for 30% of enterprise users",
             "high priority", 0.87)
        ]
        for text, label, score in examples:
            print(f"  Text: {text[:50]}")
            print(f"  Label: {label} ({score:.3f})\n")
        return

    classifier = pipeline(
        "zero-shot-classification")

    bug_reports = [
        "Production database down, all users affected",
        "Minor typo in footer copyright year",
        "Login broken for 30% of enterprise users",
        "Cosmetic spacing issue on settings page"
    ]

    candidate_labels = [
        "critical", "high priority",
        "medium priority", "low priority"]

    for text in bug_reports:
        result = classifier(
            text, candidate_labels)
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        print(f"  Text: {text[:50]}")
        print(f"  → {top_label} ({top_score:.3f})\n")

    print("💡 Zero-shot classified bug priority!")
    print("   No training data needed!")
    print("   Works by understanding the question! 🔥")


if __name__ == "__main__":
    sentiment_analysis_demo()
    ner_demo()
    question_answering_demo()
    zero_shot_classification()
