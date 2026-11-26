"""Command-line interface for PhishGuard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Sequence

from phishguard import __version__
from phishguard.dataset import generate_benchmark_dataset, load_csv_dataset
from phishguard.metrics import evaluate_predictions
from phishguard.pipeline import PhishGuardPipeline
from phishguard.reports import (
    format_text_report,
    to_html_report,
    to_json_report,
    to_sarif_report,
)
from phishguard.server import create_server


def _load_pipeline(model_path: str | None) -> PhishGuardPipeline:
    if model_path:
        return PhishGuardPipeline.load(model_path)
    return PhishGuardPipeline.load_default()


def _read_url_list(input_path: str) -> list[str]:
    if input_path == "-":
        raw_lines = sys.stdin.read().splitlines()
        return [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]

    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        urls: list[str] = []
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames:
            field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
            url_col = field_map.get("url") or field_map.get("uri") or field_map.get("link")
            if url_col:
                for row in reader:
                    val = (row.get(url_col) or "").strip()
                    if val:
                        urls.append(val)
                return urls
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _cmd_scan(args: argparse.Namespace) -> int:
    pipe = _load_pipeline(args.model)
    result = pipe.predict_one(args.url, top_k=args.top_k)

    if args.json:
        print(to_json_report([result]))
    else:
        print(format_text_report(result, explain=args.explain))

    if args.fail_on_phish and result.verdict in ("PHISHING", "SUSPICIOUS"):
        return 2
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    args.explain = True
    return _cmd_scan(args)


def _cmd_batch(args: argparse.Namespace) -> int:
    pipe = _load_pipeline(args.model)
    urls = _read_url_list(args.input)
    results = pipe.predict_batch(urls, top_k=args.top_k)

    if args.format == "json":
        rendered = to_json_report(results)
    elif args.format == "sarif":
        rendered = to_sarif_report(results, source_uri=args.input)
    elif args.format == "html":
        rendered = to_html_report(results)
    else:
        rendered = "\n\n".join(format_text_report(r, explain=args.explain) for r in results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)

    if args.fail_on_phish and any(r.verdict in ("PHISHING", "SUSPICIOUS") for r in results):
        return 2
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    if args.dataset:
        dataset = load_csv_dataset(args.dataset)
        source_desc = f"csv:{args.dataset}"
    else:
        dataset = generate_benchmark_dataset(n_samples=args.samples, seed=args.seed)
        source_desc = f"built-in benchmark corpus (seed={args.seed}, n={args.samples})"

    pipe = PhishGuardPipeline(
        max_ngram_features=args.max_ngrams,
        l2_reg=args.l2_reg,
        max_iter=args.max_iter,
    )
    cv = pipe.cross_validate(
        dataset.urls,
        dataset.labels,
        n_splits=args.folds,
        seed=args.seed,
        max_fpr=args.max_fpr,
    )
    pipe.threshold = cv.recommended_threshold
    pipe.fit(
        dataset.urls,
        dataset.labels,
        metadata={
            "source": source_desc,
            "cv_metrics": cv.aggregate_metrics.to_dict(),
        },
    )
    saved_path = pipe.save(args.output)

    summary = {
        "model_path": str(saved_path),
        "samples": len(dataset),
        "recommended_threshold": round(cv.recommended_threshold, 6),
        "cv_aggregate": cv.aggregate_metrics.to_dict(),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        agg = cv.aggregate_metrics
        print(f"Trained PhishGuard model on {len(dataset)} URLs ({args.folds}-fold stratified CV)")
        print(
            f"CV Metrics: Accuracy={agg.accuracy:.4f} | Precision={agg.precision:.4f} | "
            f"Recall={agg.recall:.4f} | F1={agg.f1:.4f} | ROC-AUC={agg.roc_auc:.4f} | "
            f"PR-AUC={agg.pr_auc:.4f} | FPR={agg.fpr:.4f}"
        )
        print(f"Selected threshold: {cv.recommended_threshold:.4f}")
        print(f"Saved safe JSON model to: {saved_path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    pipe = _load_pipeline(args.model)
    if args.dataset:
        dataset = load_csv_dataset(args.dataset)
    else:
        dataset = generate_benchmark_dataset(n_samples=args.samples, seed=args.seed)

    probs = pipe.predict_proba(dataset.urls)
    threshold = args.threshold if args.threshold is not None else pipe.threshold
    metrics = evaluate_predictions(dataset.labels, probs, threshold=threshold)

    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2))
    else:
        cm = metrics.confusion_matrix
        print(f"Evaluated {len(dataset)} URLs at threshold={metrics.threshold:.4f}")
        print(
            f"Accuracy={metrics.accuracy:.4f} | Precision={metrics.precision:.4f} | "
            f"Recall={metrics.recall:.4f} | F1={metrics.f1:.4f} | "
            f"ROC-AUC={metrics.roc_auc:.4f} | PR-AUC={metrics.pr_auc:.4f} | FPR={metrics.fpr:.4f}"
        )
        print(f"Confusion Matrix: TP={cm.tp} FP={cm.fp} TN={cm.tn} FN={cm.fn}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    pipe = _load_pipeline(args.model)
    server = create_server(pipe, host=args.host, port=args.port)
    print(f"PhishGuard local UI running at http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down local server.")
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishguard",
        description="Explainable, offline phishing and malicious URL classifier.",
    )
    parser.add_argument("--version", action="version", version=f"phishguard {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan a single URL offline.")
    p_scan.add_argument("url", help="URL to inspect (never fetched over the network).")
    p_scan.add_argument("--model", help="Path to custom JSON model file.")
    p_scan.add_argument("--explain", action="store_true", help="Show per-feature log-odds breakdown.")
    p_scan.add_argument("--top-k", type=int, default=6, help="Number of top drivers to display.")
    p_scan.add_argument("--json", action="store_true", help="Output result as JSON.")
    p_scan.add_argument(
        "--fail-on-phish",
        action="store_true",
        help="Exit with code 2 if verdict is PHISHING or SUSPICIOUS.",
    )
    p_scan.set_defaults(func=_cmd_scan)

    # explain
    p_explain = subparsers.add_parser(
        "explain", help="Scan a URL with full per-feature log-odds explainability."
    )
    p_explain.add_argument("url", help="URL to inspect.")
    p_explain.add_argument("--model", help="Path to custom JSON model file.")
    p_explain.add_argument("--top-k", type=int, default=8, help="Number of top drivers to display.")
    p_explain.add_argument("--json", action="store_true", help="Output result as JSON.")
    p_explain.add_argument("--fail-on-phish", action="store_true", help="Exit with code 2 on threat.")
    p_explain.set_defaults(func=_cmd_explain)

    # batch
    p_batch = subparsers.add_parser(
        "batch", help="Scan a file of URLs (txt, csv, or '-' for stdin) and export reports."
    )
    p_batch.add_argument("input", help="Path to URL list (.txt/.csv) or '-' for stdin.")
    p_batch.add_argument("--model", help="Path to custom JSON model file.")
    p_batch.add_argument(
        "--format",
        choices=("text", "json", "sarif", "html"),
        default="text",
        help="Output report format.",
    )
    p_batch.add_argument("--output", "-o", help="Write report to file instead of stdout.")
    p_batch.add_argument("--explain", action="store_true", help="Include log-odds in text output.")
    p_batch.add_argument("--top-k", type=int, default=5, help="Top feature drivers per URL.")
    p_batch.add_argument(
        "--fail-on-phish",
        action="store_true",
        help="Exit with code 2 if any URL is PHISHING or SUSPICIOUS.",
    )
    p_batch.set_defaults(func=_cmd_batch)

    # train
    p_train = subparsers.add_parser(
        "train", help="Train a custom offline JSON model with stratified K-fold CV."
    )
    p_train.add_argument("--dataset", help="Path to labeled CSV (url,label). Uses benchmark if omitted.")
    p_train.add_argument("--samples", type=int, default=1600, help="Benchmark sample count.")
    p_train.add_argument("--folds", type=int, default=5, help="Stratified CV folds.")
    p_train.add_argument("--max-ngrams", type=int, default=384, help="Max character n-gram features.")
    p_train.add_argument("--l2-reg", type=float, default=0.015, help="L2 regularization strength.")
    p_train.add_argument("--max-iter", type=int, default=450, help="Max Adam optimization steps.")
    p_train.add_argument("--max-fpr", type=float, default=None, help="Optional max FPR for threshold tuning.")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_train.add_argument("--output", "-o", required=True, help="Destination JSON model file.")
    p_train.add_argument("--json", action="store_true", help="Print training summary as JSON.")
    p_train.set_defaults(func=_cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a model on a labeled CSV or benchmark set.")
    p_eval.add_argument("--dataset", help="Path to labeled CSV (url,label).")
    p_eval.add_argument("--samples", type=int, default=600, help="Benchmark sample count if no CSV.")
    p_eval.add_argument("--seed", type=int, default=1337, help="Random seed for held-out benchmark.")
    p_eval.add_argument("--model", help="Path to custom JSON model file.")
    p_eval.add_argument("--threshold", type=float, default=None, help="Override decision threshold.")
    p_eval.add_argument("--json", action="store_true", help="Output evaluation metrics as JSON.")
    p_eval.set_defaults(func=_cmd_eval)

    # serve
    p_serve = subparsers.add_parser(
        "serve", help="Run optional zero-dependency local web UI on 127.0.0.1."
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p_serve.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    p_serve.add_argument("--model", help="Path to custom JSON model file.")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
