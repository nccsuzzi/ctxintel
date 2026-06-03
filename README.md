# ctxintel — Context Operating System for LLM Applications

**ctxintel** is a fully local, zero-cost context intelligence engine for LLM applications. Instead of blindly summarizing or truncating conversations, it runs a structured 6-stage deterministic pipeline that scores, extracts, remembers, compresses, and optimizes your context window — all without a single external AI API call.

> 🔒 **Fully offline** · 🎯 **Deterministic & debuggable** · 📊 **Data-driven rules** · ⚡ **Drop-in for any LLM framework**

---

## Install

```bash
pip install ctxintel

# Optional: enable NER and dependency parsing
pip install spacy
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from ctxintel import ContextIntel

sdk = ContextIntel(preset="coding_assistant")

messages = [
    {"role": "user",      "content": "Hi, I'm John. Building a REST API."},
    {"role": "assistant", "content": "What stack are you using?"},
    {"role": "user",      "content": "FastAPI and Python. Deploying on AWS."},
    {"role": "user",      "content": "Don't use synchronous code."},
    {"role": "user",      "content": "ok"},
    {"role": "user",      "content": "cool"},
    {"role": "user",      "content": "Now add JWT authentication."},
]

result = sdk.process(messages)
print(f"Tokens: {result.original_token_count} → {result.token_count}")
print(f"Memories: {result.extracted_memories}")
print(sdk.memory_summary())
```

## How It Works

```
Raw Messages
     │
     ▼
┌─────────────────┐
│  1. RANKING     │  TF-IDF + recency + signal patterns + semantic uniqueness
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. EXTRACTION  │  patterns.yaml rules + spaCy NER + dependency parsing
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. MEMORY      │  Per-memory scoring + frequency tracking + JSON persistence
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. COMPRESSION │  Extractive summarization via sumy LSA (zero AI calls)
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. OPTIMIZATION│  Fit within token budget via tiktoken
└────────┬────────┘
         ▼
   Final Context
   (minimal tokens, maximum relevance)
```

## Presets

| Preset | Use Case | Token Budget | Threshold |
|---|---|---|---|
| `coding_assistant` | Code generation & programming | 8,000 | 0.35 |
| `customer_support` | Support bots & helpdesk | 6,000 | 0.40 |
| `ai_tutor` | Educational assistants | 10,000 | 0.30 |
| `agent_system` | Autonomous agents | 12,000 | 0.45 |
| `general` | General chat apps | 8,000 | 0.40 |

```python
from ctxintel import list_presets
print(list_presets())
```

## Extending Patterns

The core innovation of ctxintel is its data-driven rule engine at `ctxintel/data/patterns.yaml`. You can add custom categories without changing any code:

```yaml
# Add to the categories section in patterns.yaml
categories:
  database:
    patterns:
      - "using {X} database"
      - "store data in {X}"
      - "migrate to {X}"
    keywords:
      - postgresql
      - mysql
      - mongodb
      - redis
    priority: high
```

No code changes needed — ctxintel loads the YAML at runtime.

## Why Not Just Summarize?

| Approach | What You Lose |
|---|---|
| **Truncation** | Early context, user preferences, constraints |
| **Blind summarization** | Structured facts, decisions, priorities |
| **ctxintel** | Nothing important — it scores, extracts, and remembers |

ctxintel doesn't just shorten your context. It **understands** what matters: user preferences are preserved, tasks are tracked, decisions are remembered, and filler is compressed — all deterministically.

## Zero API Calls

> ⚠️ **ctxintel makes zero external API calls. Ever.**
>
> No OpenAI. No Anthropic. No Cohere. No network requests.
> Everything runs locally using scikit-learn, sumy, spaCy, and tiktoken.
> Your conversations never leave your machine.

## API Reference

### `ContextIntel`

```python
sdk = ContextIntel(
    preset="coding_assistant",  # or None for manual config
    preserve=["task", "constraint"],
    threshold=0.4,
    token_budget=8000,
    memory_path=".ctxintel_memory.json",
)

# Full pipeline
result = sdk.process(messages)

# Incremental workflow
sdk.add_message("user", "Hello!")
sdk.add_message("assistant", "Hi there!")
result = sdk.flush()

# Utilities
sdk.preview_compression(messages)
sdk.memory_summary()
sdk.reset_memory()
sdk.supported_categories()
```

### `ContextResult`

```python
result.messages             # Final optimized messages
result.memories             # Extracted memories
result.token_count          # Final token count
result.original_token_count # Original token count
result.compression_ratio    # Reduction ratio (0.0–1.0)
result.extracted_memories   # Number of memories found
```

## Roadmap

- [ ] Sentence-transformers reranker (opt-in)
- [ ] CLI: `ctxintel compress chat.json --budget 4000`
- [ ] Streaming message support
- [ ] Custom patterns.yaml path support
- [ ] LangChain / LlamaIndex integration wrappers
- [ ] Multi-conversation memory aggregation

## License

MIT
