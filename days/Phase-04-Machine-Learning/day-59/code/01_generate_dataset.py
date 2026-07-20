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
