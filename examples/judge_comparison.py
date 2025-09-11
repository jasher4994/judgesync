"""Example of comparing multiple judges to find the best alignment."""

import logging

from judgesync import AlignmentTracker, ScoreRange

# Configure logging - only show INFO and above for judgesync
logging.basicConfig(
    level=logging.WARNING,  # Set default to WARNING
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# Set judgesync to INFO level to see our messages
logging.getLogger("judgesync").setLevel(logging.INFO)

# Suppress other verbose loggers
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load your evaluation data
tracker = AlignmentTracker(score_range=ScoreRange.FIVE_POINT)
tracker.load_human_scores_from_csv("examples/sample_data.csv")

# Method 1: Compare different prompts
print("=== COMPARING DIFFERENT PROMPTS ===")
prompt_comparison = tracker.create_comparison()

prompt_comparison.add_judge(
    name="strict",
    system_prompt="You are a very strict evaluator. Only give high scores to exceptional responses.",
)

prompt_comparison.add_judge(
    name="balanced",
    system_prompt="You are a balanced evaluator. Consider both strengths and weaknesses fairly.",
)

prompt_comparison.add_judge(
    name="lenient",
    system_prompt="You are a generous evaluator. Focus on the positive aspects of responses.",
)

prompt_comparison.add_judge(
    name="rubric",
    system_prompt="""Rate based on these criteria:
                     - Accuracy (40%)
                     - Completeness (30%)
                     - Clarity (30%)""",
)

# Run comparison
prompt_results = prompt_comparison.run_comparison(
    tracker.data_loader.items, use_async=True
)
print(prompt_results)

# Visualize prompt comparison
prompt_comparison.plot_comparison(
    prompt_results, save_path="prompt_comparison.png", show=False
)
print("Prompt comparison saved to prompt_comparison.png")

# Method 2: Compare different model configurations
print("\n=== COMPARING DIFFERENT MODEL CONFIGS ===")
model_comparison = tracker.create_comparison()

model_comparison.add_judge(
    name="gpt-4-cold",
    system_prompt="Evaluate the response quality.",
    temperature=0.0,
)

model_comparison.add_judge(
    name="gpt-4-warm",
    system_prompt="Evaluate the response quality.",
    temperature=0.7,
)

# Run comparison
model_results = model_comparison.run_comparison(
    tracker.data_loader.items, use_async=True
)
print(model_results)

# Visualize model comparison
model_comparison.plot_comparison(
    model_results, save_path="model_comparison.png", show=False
)
print("Model comparison saved to model_comparison.png")

# Analyze disagreements for prompts
disagreements = prompt_comparison.get_disagreement_items(prompt_results, threshold=1.0)
print(f"\nFound {len(disagreements)} items with high disagreement between prompts")
