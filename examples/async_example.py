"""Example showing async batch processing for faster scoring."""

import time

from judgesync import AlignmentTracker, ScoreRange

tracker = AlignmentTracker(score_range=ScoreRange.FIVE_POINT)

tracker.load_human_scores_from_csv("examples/sample_data.csv")

tracker.set_judge(
    system_prompt="You are an expert evaluator. Rate the response quality from 1-5."
)

print("Testing performance with 50 items...")

# Sync mode (slow)
start = time.time()
tracker.judge.score_items(tracker.data_loader.items[:50], use_async=False)
sync_time = time.time() - start
print(f"Sync scoring: {sync_time:.1f} seconds")

start = time.time()
tracker.judge.score_items_async(tracker.data_loader.items[:50], batch_size=10)
async_time = time.time() - start
print(f"Async scoring: {async_time:.1f} seconds")

print(f"Speedup: {sync_time/async_time:.1f}x faster!")
