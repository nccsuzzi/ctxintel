"""A long conversation example demonstrating ctxintel compression and memory at scale."""

from ctxintel import ContextIntel

def main():
    messages = [
        {"role": "system", "content": "You are an expert Python developer."},
        {"role": "user", "content": "Here is a bunch of old legacy code I found that we might want to reference, but probably not use exactly:\n" + ("def legacy_auth_check():\n    # checking things\n    pass\n" * 20)},
        {"role": "assistant", "content": "I see the legacy code. What do you want to do with it?"},
        {"role": "user", "content": "Hi, I'm Sarah. I need to build a new microservice. We can ignore the legacy code entirely."},
        {"role": "assistant", "content": "Hello Sarah! I can help you build that microservice. What language and framework are you thinking of using?"},
        {"role": "user", "content": "I prefer Python. Let's use FastAPI since it is fast."},
        {"role": "assistant", "content": "FastAPI is a great choice for Python microservices. Will this be deployed to the cloud?"},
        {"role": "user", "content": "Yes, we are deploying to GCP. We'll use Cloud Run."},
        {"role": "assistant", "content": "GCP Cloud Run works perfectly with FastAPI. Do you have any specific constraints or database requirements?"},
        {"role": "user", "content": "Everything must be async. Don't use any synchronous blocking calls. Also, no ORMs, we decided to use raw SQL with asyncpg."},
        {"role": "assistant", "content": "I've noted that everything must be async, no ORMs, and you're using asyncpg."},
        {"role": "user", "content": "Now add JWT authentication to the API."},
        {"role": "assistant", "content": "Sure, I will add JWT authentication to the FastAPI service using PyJWT and fastapi.security.OAuth2PasswordBearer."},
        {"role": "user", "content": "Can you also explain how asyncpg handles connection pooling?"},
        {"role": "assistant", "content": "asyncpg provides a highly efficient connection pool via `asyncpg.create_pool()`. It manages a pool of database connections and allows you to acquire and release connections asynchronously using `async with pool.acquire() as conn:`. This ensures that your FastAPI endpoints don't block while waiting for a database connection to open, which is essential for high concurrency."},
        {"role": "user", "content": "thanks, that makes sense."},
        {"role": "user", "content": "ok, perfect."},
        {"role": "user", "content": "I need to implement a health check endpoint next."},
    ]

    sdk = ContextIntel(preset="coding_assistant", threshold=0.60)

    print("=" * 60)
    print("  ctxintel — Long Conversation Example")
    print("=" * 60)

    print("\n=== Processing ===")
    result = sdk.process(messages)

    print(f"  Original tokens   : {result.original_token_count}")
    print(f"  Tokens saved      : -{result.tokens_saved}   (removed filler & compressed)")
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
        preview_text = msg.content.replace("\n", " ")
        if len(preview_text) > 120:
            preview_text = preview_text[:120] + "..."
        print(f"  [{msg.role:<10}] (imp={msg.importance:.2f}) {preview_text}")

if __name__ == "__main__":
    main()
