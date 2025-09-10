"""Command-line interface for JudgeSync."""

import argparse

from judgesync import AlignmentTracker, ScoreRange


def main():
    parser = argparse.ArgumentParser(
        description="Align LLM judges to human preferences"
    )
    parser.add_argument("csv_file", help="Path to CSV file with human scores")
    parser.add_argument("--prompt", required=True, help="System prompt for judge")
    parser.add_argument(
        "--score-range",
        default="FIVE_POINT",
        choices=["BINARY", "FIVE_POINT", "PERCENTAGE", "TEN_POINT"],
    )
    args = parser.parse_args()

    tracker = AlignmentTracker(score_range=getattr(ScoreRange, args.score_range))
    tracker.load_human_scores_from_csv(args.csv_file)
    tracker.set_judge(args.prompt)

    tracker.run_alignment_test()
    print(tracker.summary())


if __name__ == "__main__":
    main()
