import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

warnings.filterwarnings("ignore", category=DeprecationWarning, module="chromadb")

os.environ.setdefault("VOYAGE_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
