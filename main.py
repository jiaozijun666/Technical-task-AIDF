import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Baselines that do not require external APIs
SUPPORTED_BASELINES = ["perplexity", "p_true", "mars", "mars_se"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run multiple baselines via src.generate and collect metrics."
    )
    ap.add_argument(
        "--baselines",
        default="all",
        help="Comma list or 'all'. Supported: perplexity,p_true,mars,mars_se",
    )
    ap.add_argument(
        "--data",
        default="data/val_300.jsonl",
        help="Input JSONL with fields: question, context (optional), gold",
    )
    ap.add_argument(
        "--out_dir",
        default="results",
        help="Directory to write outputs",
    )
    ap.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "ollama"],
        help="Backend used by src.generate",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Model id (HF repo id or ollama tag)",
    )
    ap.add_argument(
        "--api_base",
        default=None,
        help="Ollama API base, e.g. http://127.0.0.1:11434 (ignored for hf)",
    )
    ap.add_argument(
        "--with_context",
        action="store_true",
        help="Pass --with_context through to src.generate",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Evaluate only the first N samples (0 = all)",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Max new tokens for generation",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to invoke",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running",
    )
    return ap.parse_args()

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def pick_baselines(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return SUPPORTED_BASELINES
    items = [b.strip() for b in spec.split(",") if b.strip()]
    unknown = set(items) - set(SUPPORTED_BASELINES)
    if unknown:
        raise ValueError(
            f"Unknown baselines: {sorted(unknown)}. "
            f"Supported: {SUPPORTED_BASELINES} or 'all'."
        )
    return items

def build_cmd(
    python_exe: str,
    baseline: str,
    args: argparse.Namespace,
    out_path: Path,
) -> List[str]:
    cmd = [
        python_exe, "-m", "src.generate",
        "--baseline", baseline,
        "--backend", args.backend,
        "--model", args.model,
        "--data", args.data,
        "--out", str(out_path),
        "--max_new_tokens", str(args.max_new_tokens),
        "--limit", str(args.limit),
        "--temperature", str(args.temperature),
    ]
    if args.with_context:
        cmd.append("--with_context")
    if args.backend == "ollama" and args.api_base:
        cmd.extend(["--api_base", args.api_base])
    return cmd

def parse_auroc(text: str) -> Optional[float]:
    """
    Try to find a line containing 'AUROC' and parse the last token as float.
    Works with logs printed by src.generate.
    """
    for line in text.splitlines():
        if "AUROC" in line:
            tokens = line.strip().split()
            try:
                return float(tokens[-1])
            except Exception:
                continue
    return None

def run_one(
    baseline: str,
    args: argparse.Namespace,
    out_dir: Path
) -> Tuple[str, Optional[float], Optional[str], Optional[str]]:
    """
    Run a single baseline via subprocess. Returns:
    (baseline, auroc, out_file, error_msg)
    """
    out_file = out_dir / f"val_{baseline}.jsonl"
    cmd = build_cmd(args.python, baseline, args, out_file)

    print(f"\n=== [{baseline}] ===")
    print("CMD:", shlex.join(cmd))
    if args.dry_run:
        return baseline, None, str(out_file), None

    env = os.environ.copy()
    # Ensure this repo is importable by src.generate
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + "\n" + stderr

    if proc.returncode != 0:
        print(combined)
        return baseline, None, None, f"returncode={proc.returncode}"

    auroc = parse_auroc(combined)
    if auroc is None:
        print(combined)
        print("[WARN] AUROC not found in logs.")

    return baseline, auroc, str(out_file), None

def main() -> None:
    args = parse_args()
    baselines = pick_baselines(args.baselines)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    summary: List[Dict[str, Any]] = []
    for b in baselines:
        base, auroc, out_file, err = run_one(b, args, out_dir)
        summary.append(
            {"baseline": base, "auroc": auroc, "out": out_file, "error": err}
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for row in summary:
        print(
            f"{row['baseline']:>10}  "
            f"AUROC={row['auroc']}  "
            f"out={row['out']}  "
            f"error={row['error']}"
        )
    print(f"\nSaved: {summary_path}")

if __name__ == "__main__":
    main()
