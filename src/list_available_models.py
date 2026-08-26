"""
Lists the models actually available on your Groq account. Run this
whenever you get a "model not found" error instead of guessing another
name -- model availability on free tiers varies by account/region and
changes over time (see PRD bug diary / README known limitations).

Usage (from the repo root):
    python src/list_available_models.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

try:
    from groq import Groq
except ImportError:
    print("The groq package isn't installed. Run: pip install groq")
    sys.exit(1)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("GROQ_API_KEY is not set in your .env file.")
    sys.exit(1)

client = Groq(api_key=api_key)

try:
    models = client.models.list()
except Exception as e:
    print(f"Couldn't list models: {e}")
    sys.exit(1)

print("Models available on your Groq account:\n")
for m in sorted(models.data, key=lambda m: m.id):
    active = getattr(m, "active", None)
    status = "" if active in (None, True) else "  (inactive)"
    print(f"  {m.id}{status}")

print("\nSet GROQ_MODEL in your .env to one of the IDs above.")