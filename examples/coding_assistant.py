"""Coding assistant example — demonstrates ctxintel with a realistic dev conversation."""

from ctxintel import ContextIntel


def main():
    messages = [
        {"role": "user",      "content": "Hi, I'm Usman. I'm building a REST API."},
        {"role": "assistant", "content": "Great! What stack are you using?"},
        {"role": "user",      "content": "FastAPI and Python. Deploying on AWS ECS."},
        {"role": "assistant", "content": "Good choice. Any constraints?"},
        {"role": "user",      "content": "Everything must be async. No sync code at all."},
        {"role": "user",      "content": "We're following clean architecture."},
        {"role": "assistant", "content": "Understood. I'll keep that in mind."},
        {"role": "user",      "content": "Also we decided to use PostgreSQL."},
        {"role": "assistant", "content": "PostgreSQL is solid for this use case."},
        {"role": "user",      "content": "ok"},
        {"role": "user",      "content": "cool"},
        {"role": "user",      "content": "Can you explain dependency injection?"},
        {"role": "assistant", "content": "Dependency injection is a design pattern where dependencies are provided to a class rather than created internally. In FastAPI, you use Depends() to inject services, database sessions, and authentication into your route handlers."},
        {"role": "user",      "content": "ok got it"},
        {"role": "user",      "content": "What about async SQLAlchemy?"},
        {"role": "assistant", "content": "Use asyncpg with SQLAlchemy 2.0. Define your engine with create_async_engine and use AsyncSession for all database operations."},
        {"role": "user",      "content": "nice"},
        {"role": "user",      "content": "We also need Redis for caching."},
        {"role": "assistant", "content": "Redis integrates well with FastAPI via aioredis."},
        {"role": "user",      "content": "Now add JWT authentication to the API."},
    ]

    sdk = ContextIntel(preset="coding_assistant")

    print("=" * 60)
    print("  ctxintel — Coding Assistant Example")
    print("=" * 60)

    print("\n=== Compression Preview ===")
    preview = sdk.preview_compression(messages)
    for k, v in preview.items():
        print(f"  {k}: {v}")

    print("\n=== Processing ===")
    result = sdk.process(messages)

    print(f"  Original tokens   : {result.original_token_count}")
    print(f"  Tokens saved      : -{result.tokens_saved}   (removed filler)")
    print(f"  Context injected  : +{result.context_injected}  (added memory)")
    print(f"  Final tokens      : {result.token_count}")
    print(f"  Reduction         : {result.compression_ratio * 100:.0f}%")
    print(f"  Memories found    : {result.extracted_memories}")

    print("\n=== Memory Summary ===")
    summary = sdk.memory_summary()
    if summary:
        print(summary)
    else:
        print("  (no high-importance memories)")

    print("\n=== Final Context Messages ===")
    for msg in result.messages:
        preview_text = msg.content[:80].replace("\n", " ")
        print(f"  [{msg.role:<10}] (imp={msg.importance:.2f}) {preview_text}")

    # Cleanup
    sdk.reset_memory()


if __name__ == "__main__":
    main()
