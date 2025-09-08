"""Basic usage example for JudgeSync."""

from judgesync import AlignmentTracker, ScoreRange

def main():
    # Initialize the tracker with 5-point scale
    tracker = AlignmentTracker(score_range=ScoreRange.FIVE_POINT)
    
    # Add some evaluation items manually
    tracker.add_evaluation_item(
        question="What is the capital of France?",
        response="The capital of France is Paris.",
        human_score=5
    )
    
    tracker.add_evaluation_item(
        question="What is 2+2?",
        response="2+2 equals 4.",
        human_score=5
    )
    
    tracker.add_evaluation_item(
        question="Explain quantum physics",
        response="Quantum physics is complicated.",
        human_score=2
    )
    
    # Set up the judge with a system prompt
    system_prompt = """You are an expert evaluator. Rate the quality of responses on a scale of 1-5:
    5 = Excellent, complete, and accurate
    4 = Good, mostly complete
    3 = Adequate, partially complete
    2 = Poor, incomplete or partially incorrect
    1 = Very poor, incorrect or irrelevant
    
    Consider accuracy, completeness, and clarity in your evaluation."""
    
    tracker.set_judge(system_prompt)
    
    # Run the alignment test
    results = tracker.run_alignment_test()
    
    # Print results
    print(tracker.summary())
    print(f"\nDetailed Results:\n{results}")
    
    # Export the prompt
    tracker.export_prompt("best_prompt.txt")

if __name__ == "__main__":
    main()