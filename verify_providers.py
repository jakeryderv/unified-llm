"""Quick verification that all configured providers work end-to-end.

All cloud providers (OpenAI, Anthropic, Google) and Ollama are backed by LiteLLM.
API keys are read from standard environment variables by LiteLLM automatically.
"""

import os
import sys

from unified_llm import UnifiedLLM, GenerationConfig

PROMPT = "What is 2+2? Reply with just the number."
CONFIG = GenerationConfig(temperature=0, max_tokens=32)

def check(label: str, fn):
    try:
        resp = fn()
        print(f"  [OK]  {label}: {resp.content.strip()[:80]}")
        print(f"        tokens={resp.usage.total_tokens}  latency={resp.latency_ms:.0f}ms")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


llm = UnifiedLLM()
results = {}

# --- Ollama ---
print("\n--- Ollama ---")
try:
    llm.add_provider("ollama")
    results["ollama"] = check("ollama/qwen3:4b", lambda: llm.complete("ollama/qwen3:4b", PROMPT, CONFIG))
except Exception as e:
    print(f"  [FAIL] setup: {e}")
    results["ollama"] = False

# --- OpenAI ---
print("\n--- OpenAI ---")
if os.environ.get("OPENAI_API_KEY"):
    try:
        llm.add_provider("openai")
        results["openai"] = check("openai/gpt-4o", lambda: llm.complete("openai/gpt-4o", PROMPT, CONFIG))
    except Exception as e:
        print(f"  [FAIL] setup: {e}")
        results["openai"] = False
else:
    print("  [SKIP] OPENAI_API_KEY not set")
    results["openai"] = None

# --- Anthropic ---
print("\n--- Anthropic ---")
if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        llm.add_provider("anthropic")
        results["anthropic"] = check("anthropic/claude-sonnet-4-20250514", lambda: llm.complete("anthropic/claude-sonnet-4-20250514", PROMPT, CONFIG))
    except Exception as e:
        print(f"  [FAIL] setup: {e}")
        results["anthropic"] = False
else:
    print("  [SKIP] ANTHROPIC_API_KEY not set")
    results["anthropic"] = None

# --- Google ---
print("\n--- Google ---")
if os.environ.get("GOOGLE_API_KEY"):
    try:
        llm.add_provider("google")
        results["google"] = check("google/gemini-2.5-flash", lambda: llm.complete("google/gemini-2.5-flash", PROMPT, CONFIG))
    except Exception as e:
        print(f"  [FAIL] setup: {e}")
        results["google"] = False
else:
    print("  [SKIP] GOOGLE_API_KEY not set")
    results["google"] = None

# --- Summary ---
llm.shutdown()
print("\n--- Summary ---")
passed = sum(1 for v in results.values() if v is True)
failed = sum(1 for v in results.values() if v is False)
skipped = sum(1 for v in results.values() if v is None)
print(f"  {passed} passed, {failed} failed, {skipped} skipped")
sys.exit(1 if failed else 0)
