import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
