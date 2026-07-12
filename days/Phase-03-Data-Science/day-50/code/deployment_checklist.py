"""
Day 50 — Deployment Checklist
Topic: Production readiness checks
Date: 07 July 2026
Author: Bala Ravi
"""
import os
import sys


def check_file_exists(path: str,
                       required: bool = True) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "✅" if exists else (
        "❌ MISSING" if required else "⚠️ Optional")
    print(f"  {status} {path}")
    return exists


def check_imports() -> bool:
    """Check all required imports."""
    print("\n📦 Checking dependencies...")
    packages = [
        'fastapi', 'uvicorn', 'pandas',
        'numpy', 'scipy', 'sklearn',
        'matplotlib', 'seaborn', 'joblib'
    ]

    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg if pkg != 'sklearn'
                       else 'sklearn')
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — NOT INSTALLED!")
            all_ok = False

    return all_ok


def check_project_structure() -> bool:
    """Check project files exist."""
    print("\n📁 Checking project structure...")

    base = "projects/indian_job_market_analyzer"
    required_files = [
        f"{base}/app.py",
        f"{base}/requirements.txt",
        f"{base}/render.yaml",
        f"{base}/src/__init__.py",
        f"{base}/src/data_collector.py",
        f"{base}/src/data_cleaner.py",
        f"{base}/src/analyzer.py",
        f"{base}/src/salary_predictor.py",
        f"{base}/src/visualizer.py",
        f"{base}/templates/index.html"
    ]

    all_exist = all(
        check_file_exists(f)
        for f in required_files)

    return all_exist


def check_api_health() -> bool:
    """Check if API is running."""
    print("\n🌐 Checking API health...")
    try:
        import requests
        res = requests.get(
            "http://localhost:8001/health",
            timeout=3)
        if res.status_code == 200:
            data = res.json()
            print(f"  ✅ API is running!")
            print(f"  ✅ Jobs loaded: "
                  f"{data.get('jobs_loaded', '?')}")
            return True
        else:
            print(f"  ❌ API returned "
                  f"{res.status_code}")
            return False
    except Exception:
        print(f"  ⚠️ API not running locally")
        print(f"     Run: uvicorn app:app "
              f"--reload --port 8001")
        return False


def print_deployment_commands() -> None:
    """Print commands to deploy."""
    print("\n🚀 Deployment Commands:")
    print()
    print("  # 1. Navigate to project")
    print("  cd projects/indian_job_market_analyzer")
    print()
    print("  # 2. Test locally")
    print("  uvicorn app:app --reload --port 8001")
    print("  # Open http://localhost:8001")
    print()
    print("  # 3. Push to GitHub")
    print("  git add .")
    print('  git commit -m "Deploy: '
          'Indian Job Market Analyzer v1.0"')
    print("  git push")
    print()
    print("  # 4. Render Setup")
    print("  → Go to render.com")
    print("  → New + → Web Service")
    print("  → Connect GitHub repo")
    print("  → Root dir: "
          "projects/indian_job_market_analyzer")
    print("  → Build: pip install -r requirements.txt")
    print("  → Start: uvicorn app:app "
          "--host 0.0.0.0 --port $PORT")
    print("  → Deploy!")
    print()
    print("  # 5. After deployment")
    print("  → Paste live URL in GitHub README")
    print("  → Share on LinkedIn!")
    print("  → Add to resume! 🎉")


def run_checklist() -> None:
    """Run complete deployment checklist."""
    print("🚀 Indian Job Market Analyzer")
    print("   Deployment Checklist")
    print("=" * 45)

    imports_ok = check_imports()
    structure_ok = check_project_structure()
    api_ok = check_api_health()

    print("\n" + "=" * 45)
    print("📋 Summary:")
    print(f"  Dependencies: "
          f"{'✅ OK' if imports_ok else '❌ Issues'}")
    print(f"  File Structure: "
          f"{'✅ OK' if structure_ok else '❌ Issues'}")
    print(f"  API Health: "
          f"{'✅ Running' if api_ok else '⚠️ Not running'}")

    if imports_ok and structure_ok:
        print("\n  ✅ Ready to deploy!")
        print_deployment_commands()
    else:
        print("\n  ❌ Fix issues before deploying!")


if __name__ == "__main__":
    run_checklist()
