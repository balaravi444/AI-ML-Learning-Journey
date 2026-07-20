"""
Day 59 — Software Bug Priority Predictor
Topic: Generate Realistic GitHub Issues Dataset
Date: 16 July 2026
Author: Bala Ravi

Synthetic dataset that mirrors real GitHub issues.
Features: title, description, metadata → priority label.
"""
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta

np.random.seed(42)


CRITICAL_TITLES = [
    "Production server down - all users affected",
    "Database connection pool exhausted",
    "Payment processing completely broken",
    "Authentication service returning 500",
    "API gateway timeout causing data loss",
    "Memory leak causing server crashes every 2h",
    "SSL certificate expired - site unreachable",
    "Critical security vulnerability - SQL injection",
    "Data corruption in user profiles",
    "Deployment pipeline completely broken",
    "Service outage affecting all enterprise customers",
    "Race condition causing silent data loss",
    "Out of memory error crashing app servers",
    "CDN failure - static assets not loading",
    "Webhooks silently dropping events in production"
]

HIGH_TITLES = [
    "Login fails for users with special characters in password",
    "Export to CSV produces malformed output",
    "Search returns incorrect results for unicode queries",
    "Email notifications not being delivered",
    "Dashboard charts showing wrong date range",
    "File upload fails for files larger than 10MB",
    "API rate limiting not working correctly",
    "Mobile app crashes on iOS 17",
    "Pagination broken on filtered search results",
    "OAuth2 flow failing for Google accounts",
    "Report generation timing out for large datasets",
    "Webhook delivery failing intermittently",
    "Password reset email not being sent",
    "User permissions not being enforced correctly",
    "Bulk import fails silently for 30% of records"
]

MEDIUM_TITLES = [
    "Date picker shows incorrect timezone",
    "Sort order resets after page refresh",
    "Copy button not working in Firefox",
    "Table column widths inconsistent",
    "Form validation message appears twice",
    "Dropdown options not alphabetically sorted",
    "Print layout cuts off right column",
    "Tooltip disappears too quickly",
    "Loading spinner stays visible after completion",
    "Search placeholder text not clearing on focus",
    "Breadcrumb navigation missing on mobile",
    "Settings page scroll position resets on save",
    "Graph legend overlaps chart on small screens",
    "Keyboard shortcut conflicts with browser default",
    "Avatar image not updating immediately after upload"
]

LOW_TITLES = [
    "Typo in error message on settings page",
    "Button label grammatically incorrect",
    "Footer copyright year still shows 2024",
    "Inconsistent capitalization in navigation menu",
    "Tooltip text wraps awkwardly on some resolutions",
    "Empty state illustration looks outdated",
    "Minor spacing issue between form fields",
    "Help text slightly misleading on billing page",
    "Console warning about deprecated prop",
    "Color of disabled button slightly off-brand",
    "Success toast notification fades too slowly",
    "Docs link on error page returns 404",
    "Mobile menu takes 2 taps to close sometimes",
    "Onboarding step counter shows wrong total",
    "Accessibility: missing aria-label on icon button"
]

CRITICAL_DESCS = [
    """## Bug Report

**Environment:** Production
**Severity:** P0 - Site Down

### What happened
Our production servers started throwing 500 errors at 14:32 UTC.
Currently 100% of users are affected and cannot access the application.

### Error Message
PostgreSQL ERROR: invalid page in block 2891 of relation base/16384/users
HINT: This error can be caused by data corruption
Rolling back deployment did NOT fix the issue.
Data integrity is at risk. Need immediate attention.

Affected users: ~8,000
Revenue impact: ~$50,000/hour""",
]

HIGH_DESCS = [
    """## Bug Report

**Environment:** Production
**Browser:** All browsers
**Severity:** High - Feature Broken

### Description
Users report that the CSV export feature produces malformed output
when the dataset contains more than 10,000 rows.

The file opens correctly in Excel for small exports but is unreadable
for large ones.

### Steps to Reproduce
1. Navigate to Reports > Export
2. Select date range with > 10k records
3. Click "Export to CSV"
4. Open downloaded file in Excel

### Expected
Clean CSV with all rows

### Actual
File shows garbled characters after row 9,999

### Logs
No errors in server logs. Issue seems client-side.""",

    """## Feature Broken: OAuth2 Google Login

Google OAuth2 login is failing for approximately 30% of users.
The other 70% can log in normally.

Error message shown to users:
"Authentication failed. Please try again."

Server logs show:
OAuth2Error: invalid_grant
Token has been expired or revoked
at OAuth2Client.getToken (oauth2client.js:234)
This is affecting enterprise customers who use Google Workspace.
Several have complained already.""",
]

MEDIUM_DESCS = [
    """## Minor Bug: Date Picker Timezone Issue

The date picker on the booking form shows dates in UTC instead
of the user's local timezone.

This causes confusion when users in IST (UTC+5:30) try to book
evening slots - they appear as next day.

Steps to reproduce:
1. Set your system timezone to IST
2. Go to Booking > New Appointment
3. Notice dates are shifted by 5.5 hours

No data loss occurs. Just confusing for users.
Would be nice to fix in the next sprint.""",

    """## UI: Sort order resets on page refresh

When a user sorts the data table by any column and then refreshes
the page, the sort order resets to the default (by date).

Expected: sort order should persist via URL parameter or localStorage
Actual: sort resets on every refresh

This is annoying for power users who always want to sort by name.
Not critical but comes up in user feedback regularly.""",
]

LOW_DESCS = [
    """## Minor: Typo in error message

On the Settings > Billing page, the error message when a
card is declined reads:

"Your card was been declined. Please try a different payment method."

Should be:

"Your card was declined. Please try a different payment method."

Note the extra "been" which is grammatically incorrect.
Small fix but looks unprofessional.""",

    """## Cosmetic: Copyright year outdated

The footer on all pages shows "© 2024 Company Inc."

Should be updated to "© 2026 Company Inc."

This is a minor cosmetic issue that doesn't affect functionality
but may look outdated to visitors.""",
]


def generate_issue(
        priority: str,
        issue_id: int) -> dict:
    """
    Generate a realistic GitHub issue for given priority.

    Args:
        priority: Critical / High / Medium / Low
        issue_id: Unique issue ID

    Returns:
        Dictionary representing the issue
    """
    titles = {
        'Critical': CRITICAL_TITLES,
        'High': HIGH_TITLES,
        'Medium': MEDIUM_TITLES,
        'Low': LOW_TITLES
    }
    descs = {
        'Critical': CRITICAL_DESCS,
        'High': HIGH_DESCS,
        'Medium': MEDIUM_DESCS,
        'Low': LOW_DESCS
    }

    title = np.random.choice(titles[priority])
    desc_template = np.random.choice(descs[priority])

    # Add realistic variation
    extra_sentences = {
        'Critical': [
            " This is causing a complete service outage.",
            " All users are affected immediately.",
            " Revenue loss is ongoing.",
            " Rollback did not resolve the issue.",
        ],
        'High': [
            " This affects a significant portion of users.",
            " Multiple customers have reported this.",
            " This is blocking important workflows.",
            " Please prioritize for current sprint.",
        ],
        'Medium': [
            " Would be good to fix this cycle.",
            " Users have mentioned this in feedback.",
            " Not urgent but affects UX.",
            " Fairly straightforward fix expected.",
        ],
        'Low': [
            " Low priority cosmetic fix.",
            " No functional impact.",
            " Fix when convenient.",
            " Spotted during routine review.",
        ]
    }

    desc = desc_template + np.random.choice(
        extra_sentences[priority])

    # Generate metadata
    base_date = datetime(2026, 1, 1)
    days_offset = np.random.randint(0, 180)
    filed_date = base_date + timedelta(days=days_offset)

    # Critical bugs often filed at odd hours
    if priority == 'Critical':
        hour = np.random.choice(
            [0, 1, 2, 3, 14, 15, 16, 17, 18,
             22, 23], p=[.08, .08, .08, .06,
                          .1, .1, .1, .1, .1,
                          .15, .15])
    else:
        hour = np.random.randint(7, 22)

    filed_date = filed_date.replace(hour=int(hour))

    # Feature engineering signals
    has_error_msg = (
        '```' in desc or
        'Error:' in desc or
        'error' in desc.lower())
    has_steps = (
        'Steps to' in desc or
        'reproduce' in desc.lower())
    has_code = '```' in desc
    has_version = any(
        v in desc for v in [
            'v1.', 'v2.', 'iOS', 'version',
            'v17', 'UTC'])

    label_count = {
        'Critical': np.random.randint(3, 7),
        'High': np.random.randint(2, 5),
        'Medium': np.random.randint(1, 4),
        'Low': np.random.randint(0, 3)
    }[priority]

    comment_count = {
        'Critical': np.random.randint(3, 15),
        'High': np.random.randint(1, 8),
        'Medium': np.random.randint(0, 5),
        'Low': np.random.randint(0, 3)
    }[priority]

    reporter_is_contributor = {
        'Critical': np.random.random() < 0.6,
        'High': np.random.random() < 0.4,
        'Medium': np.random.random() < 0.25,
        'Low': np.random.random() < 0.15
    }[priority]

    repos = [
        'api-gateway', 'auth-service',
        'frontend', 'payment-service',
        'data-pipeline', 'mobile-app',
        'admin-dashboard', 'notification-service'
    ]

    return {
        'issue_id': issue_id,
        'title': title,
        'description': desc,
        'priority': priority,
        'repo_name': np.random.choice(repos),
        'label_count': label_count,
        'comment_count': comment_count,
        'reporter_is_contributor': int(
            reporter_is_contributor),
        'hour_filed': hour,
        'description_length': len(desc),
        'word_count': len(desc.split()),
        'has_error_message': int(has_error_msg),
        'has_steps_to_reproduce': int(has_steps),
        'has_code_snippet': int(has_code),
        'has_version_number': int(has_version),
        'exclamation_count': desc.count('!'),
        'question_count': desc.count('?'),
        'filed_date': filed_date.strftime(
            '%Y-%m-%d %H:%M')
    }


def generate_dataset(
        n_critical: int = 150,
        n_high: int = 400,
        n_medium: int = 700,
        n_low: int = 450) -> pd.DataFrame:
    """
    Generate full dataset with realistic class distribution.

    Args:
        n_critical: Number of critical issues
        n_high: Number of high issues
        n_medium: Number of medium issues
        n_low: Number of low issues

    Returns:
        DataFrame with all issues
    """
    issues = []
    issue_id = 1

    for priority, count in [
        ('Critical', n_critical),
        ('High', n_high),
        ('Medium', n_medium),
        ('Low', n_low)
    ]:
        for _ in range(count):
            issues.append(
                generate_issue(priority, issue_id))
            issue_id += 1

    df = pd.DataFrame(issues)
    df = df.sample(frac=1, random_state=42).reset_index(
        drop=True)

    return df


def analyze_dataset(df: pd.DataFrame) -> None:
    """Print dataset statistics."""
    print("=== Dataset Overview ===\n")
    print(f"Total issues: {len(df)}")
    print(f"Features: {df.shape[1]}\n")

    print("Class Distribution:")
    dist = df['priority'].value_counts()
    for priority, count in dist.items():
        pct = count / len(df) * 100
        bar = '█' * int(pct // 2)
        print(f"  {priority:<10}: {count:>4} "
              f"({pct:.1f}%) {bar}")

    print("\nAverage feature values by priority:")
    num_cols = [
        'description_length', 'label_count',
        'comment_count', 'has_error_message',
        'has_code_snippet', 'hour_filed']

    stats = df.groupby('priority')[num_cols].mean().round(2)
    print(stats.to_string())

    print("\n💡 Key Patterns:")
    print("   Critical issues → longer descriptions, "
          "more error messages, filed at odd hours")
    print("   Low issues → short descriptions, "
          "no error messages, filed during work hours")


if __name__ == "__main__":
    print("Generating GitHub Issues Dataset...\n")

    df = generate_dataset()

    os.makedirs(
        "projects/bug_priority_predictor/data",
        exist_ok=True)
    df.to_csv(
        "projects/bug_priority_predictor/data/"
        "github_issues.csv", index=False)

    print(f"✅ Saved {len(df)} issues to "
          f"projects/bug_priority_predictor/data/"
          f"github_issues.csv\n")

    analyze_dataset(df)
