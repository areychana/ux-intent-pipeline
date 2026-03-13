import argparse
import json
import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ux_pipeline import run_pipeline
from src.intent_detector import detect_intent
from src.topic_extractor import extract_topics

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "blue":   "\033[94m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "muted":  "\033[90m",
}

def c(color, text):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def print_result(text: str, result: dict, index: int = None):
    label = f"[{index}] " if index is not None else ""
    print()
    print(c("bold", f"{label}INPUT:"))
    print(f"  {c('muted', text)}")
    print(c("bold", "INTENT:"))
    confidence = result['confidence']
    print(f"  {c('blue', result['intent'])}  {c('muted', f'(confidence: {confidence})')}")
    print(c("bold", "TOPICS:"))
    print(f"  {c('cyan', ', '.join(result['topics']) if result['topics'] else 'none detected')}")
    print(c("bold", "SUMMARY:"))
    print(f"  {c('green', result['summary'])}")
    print(c("muted", f"  {result.get('processing_time_ms', '')} ms"))
    print(c("muted", "─" * 60))


def run_single(text: str, as_json: bool):
    start = time.time()
    result = run_pipeline(text)
    result["processing_time_ms"] = round((time.time() - start) * 1000, 2)

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print_result(text, result)


def run_batch(texts: list, as_json: bool):
    print(c("yellow", f"\nRunning pipeline on {len(texts)} item(s)...\n"))

    all_results = []
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            continue
        start = time.time()
        result = run_pipeline(text)
        result["processing_time_ms"] = round((time.time() - start) * 1000, 2)
        all_results.append({"text": text, **result})

        if not as_json:
            print_result(text, result, index=i + 1)

    if as_json:
        print(json.dumps(all_results, indent=2))
        return

    if len(all_results) > 1:
        all_topics = [t for r in all_results for t in r.get("topics", [])]
        intent_counts: dict = {}
        for r in all_results:
            intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1

        print(c("bold", "\nBATCH SUMMARY"))
        print(c("bold", "Top intents:"))
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            print(f"  {c('blue', intent)}: {count}")

        print(c("bold", "All topics surfaced:"))
        unique_topics = list(dict.fromkeys(all_topics))
        print(f"  {c('cyan', ', '.join(unique_topics[:10]))}")

        texts_for_clustering = [r["text"] for r in all_results]
        if len(texts_for_clustering) >= 2:
            print(c("bold", "\nCross-batch topic clusters:"))
            try:
                clusters = extract_topics(texts_for_clustering)
                for i, cluster in enumerate(clusters):
                    print(f"  Cluster {i+1}: {c('cyan', ', '.join(cluster['keywords']))}")
                    for t in cluster["texts"][:2]:
                        print(f"    {c('muted', '-> ' + t[:70])}")
            except Exception as e:
                print(c("muted", f"  (clustering skipped: {e})"))


def main():
    parser = argparse.ArgumentParser(
        description="UX Intent Analysis Pipeline CLI tester",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Single feedback text to analyze",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Comma-separated feedback items for batch mode",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a .txt file with one feedback item per line",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted display",
    )

    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(c("yellow", f"File not found: {args.file}"))
            sys.exit(1)
        with open(args.file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        run_batch(lines, args.json)

    elif args.batch:
        items = [t.strip() for t in args.batch.split(",") if t.strip()]
        run_batch(items, args.json)

    elif args.text:
        run_single(args.text, args.json)

    else:
        print(c("bold", "\n UX Intent Analysis Pipeline"))
        print(c("muted", "Type feedback text and press Enter. Ctrl+C to exit.\n"))
        try:
            while True:
                text = input(c("cyan", "feedback> ")).strip()
                if not text:
                    continue
                run_single(text, args.json)
        except KeyboardInterrupt:
            print(c("muted", "\n\nExiting."))


if __name__ == "__main__":
    main()
