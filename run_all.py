"""
run_all.py

Master script — trains and evaluates all models sequentially.
Run this before sleeping. Everything logs to WandB automatically.

Usage:
    python run_all.py

To run only specific experiments:
    python run_all.py --experiments gold hybrid
    python run_all.py --sizes 0.5b 1.5b
    python run_all.py --eval_only
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────

EXPERIMENTS = [
    # ── Gold GraphRAG (existing configs) ────────────────────────────────────
    {
        "name":        "0.5B GraphRAG Gold",
        "train_config": "configs/lora_0.5b_graphrag_gold.yaml",
        "eval_model":  "checkpoints/qwen2.5-0.5b-graphrag-gold",
        "eval_mode":   "graphrag",
        "run_name":    "qwen2.5-0.5b-graphrag-gold",
        "experiment":  "gold",
        "size":        "0.5b",
    },
    {
        "name":        "1.5B GraphRAG Gold",
        "train_config": "configs/lora_1.5b_graphrag_gold.yaml",
        "eval_model":  "checkpoints/qwen2.5-1.5b-graphrag-gold",
        "eval_mode":   "graphrag",
        "run_name":    "qwen2.5-1.5b-graphrag-gold",
        "experiment":  "gold",
        "size":        "1.5b",
    },
    {
        "name":        "3B GraphRAG Gold",
        "train_config": "configs/lora_3b_graphrag_gold.yaml",
        "eval_model":  "checkpoints/qwen2.5-3b-graphrag-gold",
        "eval_mode":   "graphrag",
        "run_name":    "qwen2.5-3b-graphrag-gold",
        "experiment":  "gold",
        "size":        "3b",
    },
    # ── Hybrid teacher evidence ──────────────────────────────────────────────
    {
        "name":        "0.5B GraphRAG Hybrid",
        "train_config": "configs/lora_0.5b_graphrag_hybrid.yaml",
        "eval_model":  "checkpoints/qwen2.5-0.5b-graphrag-hybrid",
        "eval_mode":   "graphrag",
        "run_name":    "qwen2.5-0.5b-graphrag-hybrid",
        "experiment":  "hybrid",
        "size":        "0.5b",
    },
    {
        "name":        "1.5B GraphRAG Hybrid",
        "train_config": "configs/lora_1.5b_graphrag_hybrid.yaml",
        "eval_model":  "checkpoints/qwen2.5-1.5b-graphrag-hybrid",
        "eval_mode":   "graphrag",
        "run_name":    "qwen2.5-1.5b-graphrag-hybrid",
        "experiment":  "hybrid",
        "size":        "1.5b",
    },
    {
        "name":        "3B GraphRAG Hybrid",
        "train_config": "configs/lora_3b_graphrag_hybrid.yaml",
        "eval_model":  "checkpoints/qwen2.5-3b-graphrag-hybrid",
        "eval_mode":   "graphrag",
        "run_name":    "qwen2.5-3b-graphrag-hybrid",
        "experiment":  "hybrid",
        "size":        "3b",
    },
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

LOG_FILE = Path("run_all.log")


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], step_name: str) -> bool:
    """Run a command, stream output, return True if success."""
    log(f"START: {step_name}")
    log(f"CMD: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, text=True)

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if result.returncode == 0:
        log(f"DONE: {step_name} — {minutes}m {seconds}s")
        return True
    else:
        log(f"FAILED: {step_name} — exit code {result.returncode}")
        return False


def checkpoint_exists(path: str) -> bool:
    p = Path(path)
    return p.exists() and any(p.iterdir())


def eval_result_exists(run_name: str) -> bool:
    return Path(f"results/{run_name}/eval_results.json").exists()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    log("=" * 60)
    log("Pocket GraphRAG — Full Training + Evaluation Run")
    log("=" * 60)

    # Filter experiments
    experiments = EXPERIMENTS
    if args.experiments:
        experiments = [e for e in experiments if e["experiment"] in args.experiments]
    if args.sizes:
        experiments = [e for e in experiments if e["size"] in args.sizes]

    if not experiments:
        log("No experiments matched filters. Check --experiments and --sizes.")
        return

    log(f"Running {len(experiments)} experiments:")
    for e in experiments:
        log(f"  - {e['name']}")

    results_summary = []
    total_start = time.time()

    for exp in experiments:
        log(f"\n{'-'*60}")
        log(f"EXPERIMENT: {exp['name']}")
        log(f"{'-'*60}")

        train_success = True

        # ── Training ──────────────────────────────────────────────────────────
        if not args.eval_only:
            config = exp["train_config"]

            if not Path(config).exists():
                log(f"SKIP TRAIN: config not found — {config}")
                train_success = False
            elif checkpoint_exists(exp["eval_model"]) and not args.retrain:
                log(f"SKIP TRAIN: checkpoint exists — {exp['eval_model']}")
                log("  (use --retrain to force retrain)")
            else:
                train_success = run_cmd(
                    ["llamafactory-cli", "train", config],
                    f"Train {exp['name']}"
                )
        else:
            log("SKIP TRAIN: --eval_only mode")

        # ── Evaluation ────────────────────────────────────────────────────────
        if not train_success:
            log(f"SKIP EVAL: training failed for {exp['name']}")
            results_summary.append({
                "name": exp["name"],
                "status": "TRAIN FAILED",
            })
            continue

        if not checkpoint_exists(exp["eval_model"]):
            log(f"SKIP EVAL: checkpoint not found — {exp['eval_model']}")
            results_summary.append({
                "name": exp["name"],
                "status": "NO CHECKPOINT",
            })
            continue

        if eval_result_exists(exp["run_name"]) and not args.reeval:
            log(f"SKIP EVAL: results exist — results/{exp['run_name']}/eval_results.json")
            log("  (use --reeval to force re-evaluation)")
            results_summary.append({
                "name": exp["name"],
                "status": "ALREADY EVALUATED",
            })
            continue

        eval_cmd = [
            sys.executable,
            "src/evaluation/evaluate_student.py",
            "--model_path",  exp["eval_model"],
            "--mode",        exp["eval_mode"],
            "--run_name",    exp["run_name"],
            "--n_samples",   str(args.n_samples),
            "--output_dir",  f"results/{exp['run_name']}",
            "--examples_limit", str(args.examples_limit),
        ]
        if args.save_context:
            eval_cmd.append("--save_context")
        if args.trace_output:
            eval_cmd.append("--trace_output")

        eval_success = run_cmd(
            eval_cmd,
            f"Evaluate {exp['name']}"
        )

        results_summary.append({
            "name":   exp["name"],
            "status": "DONE" if eval_success else "EVAL FAILED",
        })

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)

    log(f"\n{'='*60}")
    log(f"ALL DONE — total time: {total_min} minutes")
    log(f"{'='*60}")
    for r in results_summary:
        log(f"  {r['status']:<20} {r['name']}")

    log("\nCompiling results table...")
    run_cmd(
        [sys.executable, "src/evaluation/compile_results.py",
         "--save_csv", "results/comparison.csv"],
        "Compile results"
    )

    log(f"\nFull log saved → {LOG_FILE}")
    log("Check WandB for training curves: https://wandb.ai")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all training and evaluation")

    parser.add_argument("--experiments", nargs="+",
                        choices=["gold", "hybrid"],
                        help="Which experiments to run (default: all)")
    parser.add_argument("--sizes", nargs="+",
                        choices=["0.5b", "1.5b", "3b"],
                        help="Which model sizes to run (default: all)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain even if checkpoint already exists")
    parser.add_argument("--reeval", action="store_true",
                        help="Re-evaluate even if results already exist")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Test samples per hop for evaluation")
    parser.add_argument("--examples_limit", type=int, default=100,
                        help="Examples to save per run. Use 0 to save all.")
    parser.add_argument("--save_context", action="store_true",
                        help="Save graph/retrieval context in eval examples for error analysis.")
    parser.add_argument("--trace_output", action="store_true",
                        help="Prompt models to emit evidence traces plus final answers during evaluation.")

    args = parser.parse_args()
    main(args)
