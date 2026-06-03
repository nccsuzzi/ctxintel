"""Agent system example — demonstrates ctxintel with a multi-step agent task."""

from ctxintel import ContextIntel


def main():
    messages = [
        {"role": "system",    "content": "You are an autonomous deployment agent."},
        {"role": "user",      "content": "Deploy the payment service to production."},
        {"role": "assistant", "content": "Starting deployment pipeline for payment service."},
        {"role": "user",      "content": "We decided to use blue-green deployment strategy."},
        {"role": "assistant", "content": "Configuring blue-green deployment on AWS ECS."},
        {"role": "user",      "content": "Don't touch the database migration until the service is healthy."},
        {"role": "assistant", "content": "Understood. I'll run health checks before migration."},
        {"role": "user",      "content": "The organization is Acme Corp and the product is PayFlow."},
        {"role": "assistant", "content": "Deploying PayFlow for Acme Corp."},
        {"role": "user",      "content": "Use Terraform for infrastructure provisioning."},
        {"role": "assistant", "content": "Terraform plan is ready. Awaiting approval."},
        {"role": "user",      "content": "Confirmed. Apply the Terraform changes."},
        {"role": "user",      "content": "ok"},
        {"role": "assistant", "content": "Terraform apply completed successfully."},
        {"role": "user",      "content": "Now run the integration tests before switching traffic."},
    ]

    sdk = ContextIntel(preset="agent_system")

    print("=" * 60)
    print("  ctxintel — Agent System Example")
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
