"""Customer support example — demonstrates ctxintel with a support conversation."""

from ctxintel import ContextIntel


def main():
    messages = [
        {"role": "user",      "content": "Hi, my name is Sarah. I need help with my order."},
        {"role": "assistant", "content": "Hello Sarah! I'd be happy to help. What's the issue?"},
        {"role": "user",      "content": "I ordered a laptop from TechCorp last week."},
        {"role": "assistant", "content": "I see. Can you provide your order number?"},
        {"role": "user",      "content": "It's ORDER-12345. The delivery was supposed to arrive yesterday."},
        {"role": "assistant", "content": "Let me check that for you."},
        {"role": "user",      "content": "I prefer email communication over phone calls."},
        {"role": "assistant", "content": "Noted. I'll make sure we communicate via email."},
        {"role": "user",      "content": "Don't send me promotional emails though."},
        {"role": "user",      "content": "Also I decided to keep the order, just need the delivery update."},
    ]

    sdk = ContextIntel(preset="customer_support")

    print("=" * 60)
    print("  ctxintel — Customer Support Example")
    print("=" * 60)

    result = sdk.process(messages)

    print(f"\n  Original tokens   : {result.original_token_count}")
    print(f"  Compressed tokens : {result.token_count}")
    print(f"  Reduction         : {result.compression_ratio * 100:.0f}%")
    print(f"  Memories found    : {result.extracted_memories}")

    print("\n=== Memory Summary ===")
    summary = sdk.memory_summary()
    if summary:
        print(summary)
    else:
        print("  (no high-importance memories)")

    print("\n=== Final Messages ===")
    for msg in result.messages:
        preview_text = msg.content[:80].replace("\n", " ")
        print(f"  [{msg.role:<10}] (imp={msg.importance:.2f}) {preview_text}")

    sdk.reset_memory()


if __name__ == "__main__":
    main()
