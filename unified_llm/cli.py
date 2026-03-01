"""CLI for unified-llm — performance profiling from the command line."""

import sys

__all__: list[str] = []

try:
    import click
except ImportError:
    print("CLI requires click: pip install click", file=sys.stderr)
    sys.exit(1)


@click.group()
@click.option("-c", "--config", "config_path", default=None, help="Path to YAML config file.")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """unified-llm — one interface to any LLM."""
    ctx.ensure_object(dict)
    if config_path:
        from unified_llm.config import create_client_from_config
        ctx.obj["client"] = create_client_from_config(config_path)
    else:
        from unified_llm.client import UnifiedLLM
        ctx.obj["client"] = UnifiedLLM()


def _ensure_provider(client, model_id: str) -> None:
    """Auto-add the provider if not already registered."""
    provider_name = model_id.split("/")[0]
    if provider_name not in client.providers:
        client.add_provider(provider_name)


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--requests", "-n", type=int, default=10, help="Number of requests per model.")
@click.option("--concurrency", type=int, default=1, help="Concurrent requests.")
@click.option("--prompt", "-p", default="Write a haiku about programming.", help="Prompt to use.")
@click.pass_context
def perf(ctx, models, requests, concurrency, prompt):
    """Run latency/throughput profiling on MODELS."""
    from unified_llm.perf import PerfConfig, PerfRunner

    client = ctx.obj["client"]
    for mid in models:
        _ensure_provider(client, mid)

    runner = PerfRunner(client)
    config = PerfConfig(num_requests=requests, concurrency=concurrency, prompt=prompt)
    report = runner.run(list(models), config)
    click.echo(report.summary_table())


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command("list")
@click.pass_context
def list_providers(ctx):
    """List registered providers and available models."""
    client = ctx.obj["client"]
    if not client.providers:
        click.echo("No providers registered. Use -c config.yaml or add providers programmatically.")
        return

    models = client.list_models()
    for provider, model_list in models.items():
        click.echo(f"\n{provider}:")
        for m in model_list:
            click.echo(f"  - {m.name}")


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

@cli.command("bench")
@click.argument("models", nargs=-1, required=True)
@click.option("--tasks", "-t", default=None, help="Comma-separated list of benchmark tasks (e.g. mmlu,gsm8k).")
@click.option("--suite", "-s", default=None, help="Preset suite: standard, safe, math, reasoning, knowledge, code.")
@click.option("--list-suites", is_flag=True, help="List available preset suites and exit.")
@click.option("--num-fewshot", "-k", type=int, default=None, help="Number of few-shot examples.")
@click.option("--limit", type=int, default=None, help="Max samples per task (for quick testing).")
@click.option("--batch-size", type=int, default=1, help="Batch size for evaluation.")
@click.option("--custom-tasks", default=None, help="Path to custom task definitions.")
@click.option("--log-samples", is_flag=True, help="Log individual samples.")
@click.option("--generate-only", is_flag=True, help="Only run generation-based tasks (no loglikelihood).")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save results to JSON file.")
@click.pass_context
def bench(ctx, models, tasks, suite, list_suites, num_fewshot, limit, batch_size, custom_tasks, log_samples, generate_only, output):
    """Run LLM benchmarks on one or more MODELS using lm-evaluation-harness."""
    try:
        from unified_llm.benchmark import BENCHMARK_SUITES, BenchmarkConfig, BenchmarkRunner, comparison_table
    except ImportError:
        click.echo("Benchmark requires lm-eval: pip install -e '.[benchmark]'", err=True)
        sys.exit(1)

    if list_suites:
        click.echo("Available benchmark suites:\n")
        for name, info in BENCHMARK_SUITES.items():
            tasks_str = ", ".join(info["tasks"])
            click.echo(f"  {name:<12} {info['description']}")
            click.echo(f"  {'':<12} Tasks: {tasks_str}\n")
        return

    if not tasks and not suite:
        click.echo("Error: specify --tasks/-t or --suite/-s (use --list-suites to see options)", err=True)
        sys.exit(1)

    client = ctx.obj["client"]
    for mid in models:
        _ensure_provider(client, mid)

    task_list = [t.strip() for t in tasks.split(",")] if tasks else []
    config = BenchmarkConfig(
        tasks=task_list,
        suite=suite,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        custom_task_paths=[custom_tasks] if custom_tasks else [],
        log_samples=log_samples,
        generate_only=generate_only,
    )

    runner = BenchmarkRunner(client)
    reports = []
    for mid in models:
        click.echo(f"\n--- Benchmarking {mid} ---\n")
        report = runner.run(mid, config)
        reports.append(report)
        click.echo(report.summary_table())

        if output and len(models) > 1:
            # Per-model output file: results.json → results_modelname.json
            base = output.rsplit(".", 1)
            safe_name = mid.replace("/", "_")
            per_model_path = f"{base[0]}_{safe_name}.{base[1]}" if len(base) > 1 else f"{output}_{safe_name}"
            report.save_json(per_model_path)
            click.echo(f"Results saved to {per_model_path}")

    # Show comparison table if multiple models
    if len(reports) > 1:
        click.echo(f"\n{'=' * 60}\n")
        click.echo(comparison_table(reports))

    if output and len(models) == 1:
        path = reports[0].save_json(output)
        click.echo(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("model")
@click.option("--system", "-s", default=None, help="System prompt.")
@click.option("--temperature", "-T", type=float, default=0.7, help="Sampling temperature.")
@click.option("--max-tokens", "-m", type=int, default=1024, help="Max tokens per response.")
@click.pass_context
def chat(ctx, model, system, temperature, max_tokens):
    """Interactive streaming chat with MODEL."""
    from unified_llm.types import GenerationConfig, Message

    client = ctx.obj["client"]
    _ensure_provider(client, model)
    provider, model_name = client._resolve(model)

    config = GenerationConfig(temperature=temperature, max_tokens=max_tokens, stream=True)
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content=system))

    click.echo(f"Chatting with {model} (Ctrl+C to quit)\n")

    try:
        while True:
            try:
                user_input = click.prompt("You", prompt_suffix="> ")
            except click.Abort:
                break

            if not user_input.strip():
                continue

            messages.append(Message(role="user", content=user_input))

            click.echo(f"\n{model_name}: ", nl=False)
            try:
                full_response = ""
                for chunk in provider.stream(messages, config, model_name):
                    click.echo(chunk.delta, nl=False)
                    full_response += chunk.delta
                click.echo("\n")
                messages.append(Message(role="assistant", content=full_response))
            except NotImplementedError:
                # Fallback to non-streaming
                response = provider.complete(messages, config, model_name)
                click.echo(response.content)
                click.echo()
                messages.append(Message(role="assistant", content=response.content))
    except (KeyboardInterrupt, EOFError):
        pass

    click.echo("\nGoodbye!")


def main():
    cli()


if __name__ == "__main__":
    main()
