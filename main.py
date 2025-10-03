from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from pathlib import Path

# ---- Global defaults ----
DEFAULTS = {
    # experiment root (where data/results will be created if you don't pass explicit paths)
    "ROOT": Path("."),

    # generation defaults
    "MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
    "WITH_CONTEXT": False,
    "TEMP": 0.5,
    "MAX_NEW_TOKENS": 64,

    # data split sizes (paper setup)
    "TRAIN_N": 2000,
    "VAL_N": 300,
    "TEST_POOL_N": 800,
    "SAMPLE_POOL_N": 500,   # test_sample_pool size

    # multi-sample (k)
    "K_SAMPLE": 6,
}

def get_paths(root: Path) -> dict:
    """Build canonical paths under a given root."""
    interim   = root / "data" / "interim"
    processed = root / "data" / "processed"
    results   = root / "results"
    return {
        "root": root,
        "interim": interim,
        "processed": processed,
        "results": results,
        # data files
        "train": interim / "train_2000.jsonl",
        "val": processed / "val_300.jsonl",
        "test_pool": interim / "test_pool_800.jsonl",
        "sample_pool": interim / "test_sample_pool_500.jsonl",
        "final_test": processed / "test_final_400.jsonl",
        # multi-sample outputs
        "samples500": results / "test500_samples.jsonl",
    }

def ensure_dirs(paths: dict) -> None:
    for p in (paths["interim"], paths["processed"], paths["results"]):
        p.mkdir(parents=True, exist_ok=True)


# --------- Baseline registry (module path -> short name) ----------
BASELINE_REGISTRY = {
    "perplexity":        "baseline.uncertainty_based.perplexity",
    "p_true":            "baseline.uncertainty_based.p_true",
    "mars":              "baseline.uncertainty_based.mars",
    "mars_se":           "baseline.uncertainty_based.mars_se",
    "se":                "baseline.uncertainty_based.semantic_entropy",
    "ccs":               "baseline.internal_representation_based.ccs",
    "haloscope":         "baseline.internal_representation_based.haloscope",
    "saplma":            "baseline.internal_representation_based.saplma",
}
SUPPORTED_BASELINES = list(BASELINE_REGISTRY.keys())


# ---------------- Utility: import-or-run-module -------------------
def import_module_or_none(dotted: str):
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None


def call_module_main_or_subprocess(module_name: str, args: List[str]) -> int:
    """
    Try to call module.main(args). If not available, fallback to:
    python -m <module_name> <args...>
    """
    mod = import_module_or_none(module_name)
    if mod is not None:
        # Prefer a function named `cli` or `main`
        entry = getattr(mod, "cli", None) or getattr(mod, "main", None)
        if entry is not None:
            return int(entry(args) or 0)
    # Fallback to subprocess
    cmd = ["python", "-m", module_name] + args
    print(f"[main] → subprocess: {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.call(cmd)


# ----------------- Utility: data loading helpers ------------------
def read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and i + 1 >= limit:
                break
    return rows


def ensure_file_exists(p: Path, desc: str = "file") -> None:
    if not p.exists():
        raise FileNotFoundError(f"{desc} not found: {p}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------- run baselines command ---------------------
def run_one_baseline(
    baseline_name: str,
    data_path: Path,
    out_dir: Path,
    model: str,
    with_context: bool,
    temperature: float,
    max_new_tokens: int,
    limit: Optional[int],
    extra: Dict[str, Any],
) -> Tuple[float, Path]:
    """
    Expect each baseline module to expose: run(samples, tokenizer, model, build_prompt, cfg, limit)
    But to keep it robust across your current files, we call their CLI if Python entry is not found.
    Baseline JSONL output path is returned together with AUROC (if provided by the module).
    """
    module_name = BASELINE_REGISTRY[baseline_name]
    split_stem = data_path.stem  # e.g., val_300
    out_path = out_dir / f"{split_stem}_{baseline_name}.jsonl"

    # Build CLI args for module fallback
    cli_args = [
        "--data", str(data_path),
        "--out", str(out_path),
        "--model", model,
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
    ]
    if with_context:
        cli_args.append("--with_context")
    if limit is not None:
        cli_args += ["--limit", str(limit)]

    # Optional knobs for semantic-entropy (SE)
    if baseline_name in ("se", "mars_se"):
        if extra.get("se_k") is not None:
            cli_args += ["--se_k", str(extra["se_k"])]
        if extra.get("se_use_gpt"):
            cli_args += ["--se_use_gpt"]
        if extra.get("se_gpt_model"):
            cli_args += ["--se_gpt_model", str(extra["se_gpt_model"])]

    print(f"[run] {baseline_name} → {out_path}")
    rc = call_module_main_or_subprocess(module_name, cli_args)
    if rc != 0:
        print(f"[run][warn] {baseline_name} exited with code {rc}. Skipping AUROC parse.")
        return -1.0, out_path

    # Try to parse AUROC if the baseline wrote a footer meta line { "auroc": x }
    auroc = -1.0
    try:
        # last non-empty line heuristic
        with out_path.open("r", encoding="utf-8") as f:
            last = ""
            for line in f:
                if line.strip():
                    last = line
        meta = json.loads(last)
        if isinstance(meta, dict) and "auroc" in meta:
            auroc = float(meta["auroc"])
    except Exception:
        pass

    return auroc, out_path


def cmd_run(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    ensure_file_exists(data_path, "data file")
    ensure_dir(out_dir)

    # Baseline selection
    if args.baselines == "all":
        baselines = SUPPORTED_BASELINES
    else:
        baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
        unknown = [b for b in baselines if b not in SUPPORTED_BASELINES]
        if unknown:
            raise ValueError(f"Unknown baseline(s): {unknown}. Supported: {SUPPORTED_BASELINES}")

    print("[run] device=auto  model=", args.model)
    print("[run] data=", data_path, f"(limit={args.limit})")
    print("[run] baselines=", baselines)
    print("[run] out_dir=", out_dir)

    extras = dict(
        se_k=args.se_k,
        se_use_gpt=args.se_use_gpt,
        se_gpt_model=args.se_gpt_model,
    )

    for b in baselines:
        auroc, path = run_one_baseline(
            baseline_name=b,
            data_path=data_path,
            out_dir=out_dir,
            model=args.model,
            with_context=args.with_context,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            extra=extras,
        )
        mark = f"{auroc:.4f}" if auroc >= 0 else "n/a"
        print(f"[run] {b} → AUROC={mark}  out={path}")


# ---------------------- prepare:* subcommands ---------------------
def cmd_prepare_random_pairs(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.out_train).parent)
    ensure_dir(Path(args.out_val).parent)
    ensure_dir(Path(args.out_test).parent)

    cli = [
        "--out-train", args.out_train,
        "--out-val",   args.out_val,
        "--out-test",  args.out_test,
        "--seed",      str(args.seed),
    ]
    rc = call_module_main_or_subprocess("src.random_pairs", cli)
    if rc != 0:
        raise SystemExit(rc)


def cmd_prepare_multi_sample(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.out).parent)
    cli = [
        "--model", args.model,
        "--data",  args.data,
        "--out",   args.out,
        "--k",     str(args.k),
        "--temperature", str(args.temperature),
        "--max_new_tokens", str(args.max_new_tokens),
    ]
    rc = call_module_main_or_subprocess("src.multi_sample", cli)
    if rc != 0:
        raise SystemExit(rc)


def cmd_prepare_final_select(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.out).parent)
    cli = [
        "--samples", args.samples,
        "--out",     args.out,
    ]
    rc = call_module_main_or_subprocess("src.final_select", cli)
    if rc != 0:
        raise SystemExit(rc)


# ----------------------------- CLI -------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Project CLI: run baselines and prepare data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Run one or more baselines.")
    pr.add_argument("--baselines", default="all",
                    help=f"Comma list or 'all'. Supported: {SUPPORTED_BASELINES}")
    pr.add_argument("--data", required=True, help="Input JSONL with {question, context?, gold}.")
    pr.add_argument("--out_dir", required=True, help="Folder for outputs.")
    pr.add_argument("--model", required=True, help="HF model id (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    pr.add_argument("--with_context", action="store_true", help="Include context in the prompt.")
    pr.add_argument("--temperature", type=float, default=0.5)
    pr.add_argument("--max_new_tokens", type=int, default=64)
    pr.add_argument("--limit", type=int, default=None, help="Eval only the first N rows (debug).")

    # SE-specific knobs
    pr.add_argument("--se_k", type=int, default=6, help="[SE] number of samples for semantic entropy.")
    pr.add_argument("--se_use_gpt", action="store_true", help="[SE] use GPT for entailment (requires API).")
    pr.add_argument("--se_gpt_model", type=str, default=None, help="[SE] GPT model name if --se_use_gpt.")

    pr.set_defaults(func=cmd_run)

    # prepare / random_pairs
    pp = sub.add_parser("prepare", help="Data preparation pipeline.")
    pps = pp.add_subparsers(dest="stage", required=True)

    prp = pps.add_parser("random_pairs", help="Create train/val/test pools.")
    prp.add_argument("--out-train", required=True)
    prp.add_argument("--out-val",   required=True)
    prp.add_argument("--out-test",  required=True)
    prp.add_argument("--seed", type=int, default=42)
    prp.set_defaults(func=cmd_prepare_random_pairs)

    pms = pps.add_parser("multi_sample", help="Generate K samples per question.")
    pms.add_argument("--model", required=True)
    pms.add_argument("--data",  required=True)
    pms.add_argument("--out",   required=True)
    pms.add_argument("--k", type=int, default=6)
    pms.add_argument("--temperature", type=float, default=0.5)
    pms.add_argument("--max_new_tokens", type=int, default=64)
    pms.set_defaults(func=cmd_prepare_multi_sample)

    pfs = pps.add_parser("final_select", help="Select the final 400 test questions (and keep val 300).")
    pfs.add_argument("--samples", required=True, help="results/test500_samples.jsonl")
    pfs.add_argument("--out",     required=True, help="data/test_final_400.jsonl")
    pfs.set_defaults(func=cmd_prepare_final_select)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
