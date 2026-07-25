"""
Day 66 — Named Entity Recognition
Topic: Regex-Based Tech Entity Extraction
Date: 23 July 2026
Author: Bala Ravi

Regex for tech entities — fast, precise, no training!
Version numbers, error codes, services, environments.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


# Tech entity regex patterns
TECH_PATTERNS = {
    'VERSION': [
        r'v\d+\.\d+(?:\.\d+)?(?:-[\w]+)?',
        r'\d+\.\d+\.\d+(?:-[\w]+)?',
        r'version\s+\d+(?:\.\d+)*'
    ],
    'ERROR_CODE': [
        r'(?:HTTP|Error|Status|Code)\s*:?\s*[45]\d{2}',
        r'(?:FATAL|CRITICAL|PANIC):\s*\w+',
        r'(?:errno|error)\s*[=:]\s*\d+',
        r'\b[45]\d{2}\s+(?:error|Error)',
        r'OOM|NullPointerException|StackOverflow'
    ],
    'SERVICE': [
        r'\b\w+(?:-\w+)*-(?:service|api|gateway|proxy|worker|job)\b',
        r'\b(?:auth|payment|notification|data|user|order|inventory)-\w+\b'
    ],
    'ENVIRONMENT': [
        r'\b(?:production|prod|staging|stage|'
        r'development|dev|testing|test|qa|'
        r'sandbox)\b'
    ],
    'CLOUD_REGION': [
        r'\b(?:us|eu|ap|sa|ca|me|af)-(?:east|west|'
        r'north|south|central|northeast|southeast)-\d\b',
        r'\b(?:eastus|westus|eastus2|centralus|'
        r'northeurope|westeurope)\b'
    ],
    'DATABASE': [
        r'\b(?:PostgreSQL|MySQL|MongoDB|Redis|'
        r'Elasticsearch|Cassandra|DynamoDB|'
        r'SQLite|MariaDB|Oracle|MSSQL)\b'
    ],
    'CLOUD_PROVIDER': [
        r'\b(?:AWS|GCP|Azure|DigitalOcean|'
        r'Heroku|Vercel|Netlify|Render)\b'
    ],
    'METRIC': [
        r'\d+(?:,\d{3})*\s*(?:users|requests|'
        r'records|events|errors|calls)',
        r'\d+(?:\.\d+)?%\s*(?:of\s+\w+)?',
        r'(?:\$|₹|€|£)\d+(?:,\d{3})*(?:\.\d+)?'
        r'(?:K|M|B)?(?:\s*(?:per\s+hour|/hr|/day))?'
    ],
    'TIME_DURATION': [
        r'since\s+(?:\d+:\d+|\w+\s+\w+)',
        r'\d+\s*(?:hours?|minutes?|seconds?|'
        r'days?|weeks?)\s+ago',
        r'(?:yesterday|last\s+\w+|this\s+morning|'
        r'since\s+\w+)'
    ]
}


class TechEntityExtractor:
    """
    Extract tech-specific entities from bug reports.

    Uses regex patterns — fast and precise!
    No training data needed.
    """

    def __init__(self) -> None:
        """Compile regex patterns."""
        self.compiled_patterns = {}
        for entity_type, patterns in (
                TECH_PATTERNS.items()):
            self.compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE)
                for p in patterns]

    def extract(self,
                text: str) -> List[Entity]:
        """
        Extract all entities from text.

        Args:
            text: Bug report or any tech text

        Returns:
            List of Entity objects
        """
        entities = []
        seen_spans = set()

        for label, patterns in (
                self.compiled_patterns.items()):
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())

                    # Skip overlapping spans
                    overlaps = False
                    for seen in seen_spans:
                        if (span[0] < seen[1] and
                                span[1] > seen[0]):
                            overlaps = True
                            break

                    if not overlaps:
                        seen_spans.add(span)
                        entities.append(Entity(
                            text=match.group().strip(),
                            label=label,
                            start=match.start(),
                            end=match.end()))

        return sorted(entities, key=lambda e: e.start)

    def extract_dict(self,
                     text: str) -> Dict[str, List[str]]:
        """
        Extract entities as dictionary.

        Args:
            text: Input text

        Returns:
            Dictionary of entity_type → [values]
        """
        entities = self.extract(text)
        result = {}
        for ent in entities:
            if ent.label not in result:
                result[ent.label] = []
            if ent.text not in result[ent.label]:
                result[ent.label].append(ent.text)
        return result


def demonstrate_extraction() -> None:
    """Show entity extraction on bug reports."""
    print("=== Tech Entity Extraction on Bug Reports ===\n")

    extractor = TechEntityExtractor()

    bug_reports = [
        ("The auth-service v2.3.1 is returning HTTP 503 "
         "errors in production on AWS us-east-1. "
         "PostgreSQL connection pool exhausted. "
         "12,000 users affected since 14:32 UTC. "
         "Revenue loss: $5,000/hr."),

        ("api-gateway v1.8.0 in staging is throwing "
         "FATAL: NullPointerException. "
         "GCP us-central1 deployment failed. "
         "Affects 30% of API calls since yesterday."),

        ("notification-service cannot connect to Redis "
         "in production environment. "
         "Error: Connection refused on port 6379. "
         "About 50,000 email notifications queued."),

        ("Minor CSS spacing issue in the dashboard UI. "
         "No service or version affected. "
         "Low priority cosmetic fix.")
    ]

    for i, report in enumerate(bug_reports, 1):
        print(f"Bug Report {i}:")
        print(f"  {report[:80]}...\n")

        entities = extractor.extract_dict(report)

        if entities:
            for entity_type, values in (
                    entities.items()):
                print(f"  [{entity_type:<18}]: "
                      f"{values}")
        else:
            print("  No tech entities detected.")
        print()


def entity_based_incident_search() -> None:
    """
    Search for similar past incidents by entity.
    """
    print("=== Entity-Based Incident Search ===\n")

    extractor = TechEntityExtractor()

    incident_db = [
        {
            'id': 'INC-001',
            'text': (
                "auth-service v2.1.0 down in production "
                "PostgreSQL connection exhausted"),
            'resolved': True,
            'fix': "Increase connection pool size to 200"
        },
        {
            'id': 'INC-002',
            'text': (
                "api-gateway HTTP 503 errors AWS us-east-1 "
                "load balancer misconfiguration"),
            'resolved': True,
            'fix': "Update load balancer health check timeout"
        },
        {
            'id': 'INC-003',
            'text': (
                "payment-service v3.2.1 FATAL error "
                "Redis connection refused production"),
            'resolved': True,
            'fix': "Restart Redis cluster, add retry logic"
        },
        {
            'id': 'INC-004',
            'text': (
                "notification-service staging "
                "MongoDB connection timeout"),
            'resolved': True,
            'fix': "Increase MongoDB timeout to 30s"
        }
    ]

    new_incident = (
        "auth-service v2.3.1 returning 503 errors "
        "in production. PostgreSQL seems overwhelmed.")

    print(f"New Incident: {new_incident}\n")

    new_entities = extractor.extract_dict(new_incident)
    print(f"Extracted entities:")
    for etype, vals in new_entities.items():
        print(f"  {etype}: {vals}")

    # Find matching past incidents
    print(f"\nSearching past incidents...\n")

    matches = []
    for inc in incident_db:
        inc_entities = extractor.extract_dict(inc['text'])
        inc_all = set(
            v.lower()
            for vals in inc_entities.values()
            for v in vals)
        new_all = set(
            v.lower()
            for vals in new_entities.values()
            for v in vals)

        overlap = inc_all & new_all
        if overlap:
            matches.append({
                'incident': inc,
                'matching_entities': overlap,
                'score': len(overlap)
            })

    matches.sort(key=lambda x: x['score'],
                 reverse=True)

    if matches:
        print("Similar past incidents:")
        for m in matches:
            inc = m['incident']
            print(f"\n  [{inc['id']}] "
                  f"Matching: {m['matching_entities']}")
            print(f"  Previous fix: {inc['fix']}")
    else:
        print("No similar incidents found.")

    print(f"\n💡 Entity-based incident matching!")
    print(f"   Same service + same error = same fix!")
    print(f"   This is what AI Engineering Copilot")
    print(f"   (Phase 6) does at company scale! 🔥")


def ner_feature_engineering() -> None:
    """
    Use NER output as ML features!
    Bug Predictor v3.
    """
    print("\n=== NER as ML Features ===\n")

    extractor = TechEntityExtractor()

    bug_reports = [
        ("auth-service v2.3.1 FATAL error production "
         "PostgreSQL down 50000 users affected",
         'Critical'),
        ("api-gateway HTTP 503 staging "
         "500 requests failing",
         'High'),
        ("dashboard CSS spacing issue development "
         "minor cosmetic",
         'Low'),
    ]

    print("NER-based feature engineering:")
    print(f"\n{'Report':<45} | "
          f"{'SERVICE':>8} | "
          f"{'ERROR':>6} | "
          f"{'PROD':>5} | "
          f"{'USERS':>6} | "
          f"{'Priority'}")
    print("-" * 85)

    for report, priority in bug_reports:
        entities = extractor.extract_dict(report)

        has_service = int(
            bool(entities.get('SERVICE')))
        has_error = int(
            bool(entities.get('ERROR_CODE')))
        is_prod = int(
            'production' in report.lower() or
            'prod' in report.lower())
        has_users = int(
            bool(entities.get('METRIC')))

        print(f"{report[:43]:<45} | "
              f"{has_service:>8} | "
              f"{has_error:>6} | "
              f"{is_prod:>5} | "
              f"{has_users:>6} | "
              f"{priority}")

    print(f"\n✅ NER features encode domain knowledge!")
    print(f"   'production' + 'FATAL' + '50000 users'")
    print(f"   = almost certainly Critical!")
    print(f"   Add these to Bug Predictor → better F1!")


if __name__ == "__main__":
    demonstrate_extraction()
    entity_based_incident_search()
    ner_feature_engineering()
