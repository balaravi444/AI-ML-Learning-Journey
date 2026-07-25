"""
Day 66 — Named Entity Recognition
Topic: spaCy NER — Built-in + Custom
Date: 23 July 2026
Author: Bala Ravi

Extract structured information from unstructured text!
spaCy makes NER incredibly easy.
"""
import re
import warnings
warnings.filterwarnings('ignore')

try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
        SPACY_AVAILABLE = True
    except OSError:
        print("⚠️  spaCy model not downloaded.")
        print("    Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    print("⚠️  spaCy not installed.")
    print("    Run: pip install spacy")
    SPACY_AVAILABLE = False


def demonstrate_spacy_ner() -> None:
    """Show spaCy built-in NER."""
    print("=== spaCy Built-in NER ===\n")

    texts = [
        ("Microsoft Azure reported a major outage in "
         "East US affecting 50,000 users since 3:00 AM UTC "
         "on July 23, 2026. Revenue loss estimated at $2M."),

        ("Google Cloud SQL instance in asia-south1 "
         "is throwing errors since yesterday afternoon. "
         "The engineering team at Flipkart noticed it first."),

        ("AWS Lambda function in us-east-1 has been "
         "timing out since the deployment on Monday. "
         "About 30% of API calls failing.")
    ]

    if not SPACY_AVAILABLE:
        print("spaCy not available — showing expected output:\n")
        expected = {
            'ORG': ['Microsoft', 'Google', 'AWS', 'Flipkart'],
            'GPE': ['East US', 'asia-south1', 'us-east-1'],
            'DATE': ['July 23, 2026', 'yesterday', 'Monday'],
            'TIME': ['3:00 AM UTC', 'afternoon'],
            'MONEY': ['$2M'],
            'PERCENT': ['30%'],
            'CARDINAL': ['50,000', '2']
        }
        for ent_type, examples in expected.items():
            print(f"  {ent_type:<12}: {examples}")
        return

    for i, text in enumerate(texts, 1):
        print(f"Text {i}: {text[:70]}...\n")
        doc = nlp(text)

        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)

        for label, values in entities.items():
            print(f"  {label:<12}: {values}")
        print()


def custom_tech_ner_demo() -> None:
    """
    Demonstrate custom NER for tech domain.
    spaCy EntityRuler for rule-based matching.
    """
    print("=== Custom Tech Entity Rules ===\n")

    if not SPACY_AVAILABLE:
        print("Showing concept — spaCy not available\n")
        print("EntityRuler pattern example:")
        print("""
patterns = [
    {"label": "SERVICE",
     "pattern": [{"LOWER": {"REGEX": r"\\w+-service"}}]},
    {"label": "ERROR_CODE",
     "pattern": [{"LOWER": "error"},
                 {"IS_DIGIT": True}]},
    {"label": "ENVIRONMENT",
     "pattern": [{"LOWER": {"IN": [
         "production", "staging", "dev"]}}]}
]
        """)
        return

    # Add custom entity ruler
    ruler = nlp.add_pipe(
        "entity_ruler", before="ner")

    patterns = [
        # Services
        {"label": "SERVICE",
         "pattern": "auth-service"},
        {"label": "SERVICE",
         "pattern": "api-gateway"},
        {"label": "SERVICE",
         "pattern": "payment-service"},
        {"label": "SERVICE",
         "pattern": "notification-service"},
        {"label": "SERVICE",
         "pattern": "data-pipeline"},
        # Environments
        {"label": "ENVIRONMENT",
         "pattern": "production"},
        {"label": "ENVIRONMENT",
         "pattern": "staging"},
        {"label": "ENVIRONMENT",
         "pattern": "development"},
        # Databases
        {"label": "DATABASE",
         "pattern": "PostgreSQL"},
        {"label": "DATABASE",
         "pattern": "MongoDB"},
        {"label": "DATABASE",
         "pattern": "Redis"},
        # Cloud
        {"label": "CLOUD",
         "pattern": "AWS"},
        {"label": "CLOUD",
         "pattern": "GCP"},
        {"label": "CLOUD",
         "pattern": "Azure"},
    ]

    ruler.add_patterns(patterns)

    test_text = (
        "The auth-service in production on AWS is "
        "failing to connect to PostgreSQL. The "
        "api-gateway is routing requests correctly "
        "but payment-service throws 503 errors.")

    doc = nlp(test_text)

    print(f"Text: {test_text}\n")
    print("Entities found:")
    for ent in doc.ents:
        print(f"  [{ent.label_:<14}] '{ent.text}'")


if __name__ == "__main__":
    demonstrate_spacy_ner()
    custom_tech_ner_demo()
