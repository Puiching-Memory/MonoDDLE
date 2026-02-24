#!/usr/bin/env python3
"""
CUDA-accelerated KITTI evaluation driver.

Usage:
    python tools/eval_cuda.py                          # Evaluate all experiments
    python tools/eval_cuda.py --exp monodle/kitti_da3  # Evaluate specific experiment
    python tools/eval_cuda.py --verify                 # Verify against train.log results

Reads pre-computed detection results from experiments/results/<exp>/outputs/data/
and evaluates against GT labels from data/KITTI/training/label_2/.

Results are compared character-by-character against the original train.log output
to ensure exact numerical equivalence.
"""
import os
import sys
import re
import time
import argparse
import logging
import pathlib
import numpy as np

# Project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import CUDA eval module
sys.path.insert(0, os.path.join(ROOT_DIR, "lib", "datasets", "kitti", "kitti_eval_cuda"))
from eval_cuda import get_official_eval_result

# Import annotation loader from original code
sys.path.insert(0, os.path.join(ROOT_DIR, "lib", "datasets", "kitti", "kitti_eval_python"))
from kitti_common import get_label_annos


def setup_logger(name="eval_cuda"):
    """Create a logger matching the train.log format."""
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(log_format))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def find_all_experiments(results_root):
    """Find all experiment directories containing outputs/data."""
    experiments = []
    for root, dirs, files in os.walk(results_root):
        if os.path.basename(root) == "data" and os.path.basename(os.path.dirname(root)) == "outputs":
            # Check there are txt files
            txt_files = [f for f in os.listdir(root) if f.endswith('.txt')]
            if txt_files:
                run_dir = os.path.dirname(os.path.dirname(root))  # outputs -> run_dir
                experiments.append(run_dir)
    return sorted(experiments)


def extract_eval_from_log(log_path):
    """Extract evaluation result lines from train.log."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Find evaluation result lines (AP lines)
    result_lines = []
    capture = False
    for line in lines:
        stripped = line.strip()
        # Extract content after log prefix "INFO  "
        match = re.search(r'INFO\s+(.*)', stripped)
        content = match.group(1) if match else stripped
        # Match header lines like "Car AP@0.70, 0.70, 0.70:" or "Car AP_R40@..."
        if re.search(r'AP@\d+\.\d+', content) or re.search(r'AP_R40@\d+\.\d+', content):
            result_lines.append(content)
            capture = True
        elif capture and re.search(r'^(bbox|bev|3d|aos)\s+AP:', content):
            result_lines.append(content)
        elif capture and content and not re.search(r'^(bbox|bev|3d|aos)\s+AP:', content):
            # Check if this is another AP header or something else
            if not (re.search(r'AP@\d+\.\d+', content) or re.search(r'AP_R40@\d+\.\d+', content)):
                capture = False
    return '\n'.join(result_lines) if result_lines else None


def parse_eval_result_str(result_str):
    """Parse evaluation result string into structured data for comparison."""
    values = []
    for line in result_str.strip().split('\n'):
        # Extract numbers from AP lines
        match = re.findall(r'[\d.]+', line)
        if match and ('AP:' in line or 'AP@' in line or 'AP_R40@' in line):
            values.extend([float(x) for x in match])
    return values


def get_val_image_ids(data_root):
    """Read val.txt image IDs."""
    val_file = os.path.join(data_root, "ImageSets", "val.txt")
    with open(val_file, 'r') as f:
        return [int(line.strip()) for line in f.readlines()]


def evaluate_experiment(run_dir, data_root, logger, class_names=None):
    """
    Evaluate a single experiment using CUDA-accelerated eval.

    Args:
        run_dir: experiment run directory (contains outputs/data/)
        data_root: KITTI data root (contains training/label_2/)
        logger: logger instance
        class_names: list of class names to evaluate (default: ['Car'])

    Returns:
        result_str: evaluation result string
        ret_dict: evaluation metrics dict
        elapsed: time in seconds
    """
    if class_names is None:
        class_names = ['Car']

    results_dir = os.path.join(run_dir, "outputs", "data")
    label_dir = os.path.join(data_root, "training", "label_2")

    logger.info("==> Loading detections and GTs...")

    # Load detection annotations (from outputs/data)
    dt_annos = get_label_annos(results_dir)

    # Load GT annotations (using val image IDs)
    img_ids = get_val_image_ids(data_root)
    gt_annos = get_label_annos(label_dir, img_ids)

    test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    logger.info("==> Evaluating (official) ...")
    start_time = time.time()

    full_result = ''
    full_ret_dict = {}
    for category in class_names:
        results_str, results_dict = get_official_eval_result(
            gt_annos, dt_annos, test_id[category])
        full_result += results_str
        full_ret_dict.update(results_dict)
        logger.info(results_str)

    elapsed = time.time() - start_time
    return full_result, full_ret_dict, elapsed


def compare_results(cuda_result, log_result, tolerance=1e-3):
    """
    Compare CUDA eval results with original log results.

    Returns: (match, details)
    """
    cuda_values = parse_eval_result_str(cuda_result)
    log_values = parse_eval_result_str(log_result)

    if len(cuda_values) != len(log_values):
        return False, f"Value count mismatch: CUDA={len(cuda_values)}, Log={len(log_values)}"

    diffs = []
    for i, (cv, lv) in enumerate(zip(cuda_values, log_values)):
        if abs(cv - lv) > tolerance:
            diffs.append(f"  idx={i}: CUDA={cv:.4f}, Log={lv:.4f}, diff={abs(cv-lv):.6f}")

    if diffs:
        return False, "Value mismatches:\n" + "\n".join(diffs)
    return True, "All values match within tolerance"


def main():
    parser = argparse.ArgumentParser(description="CUDA-accelerated KITTI Evaluation")
    parser.add_argument('--exp', type=str, default=None,
                        help='Specific experiment path (relative to experiments/results/)')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(ROOT_DIR, "data", "KITTI"),
                        help='KITTI data root directory')
    parser.add_argument('--verify', action='store_true',
                        help='Verify results against train.log')
    parser.add_argument('--classes', type=str, nargs='+', default=['Car'],
                        help='Class names to evaluate')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Tolerance for result comparison')
    args = parser.parse_args()

    logger = setup_logger()
    results_root = os.path.join(ROOT_DIR, "experiments", "results")

    if args.exp:
        # Find the specific experiment
        exp_path = os.path.join(results_root, args.exp)
        if os.path.exists(exp_path):
            # Could be the config-level dir with timestamps underneath
            experiments = find_all_experiments(exp_path)
            if not experiments:
                # Maybe it's a direct run dir
                if os.path.exists(os.path.join(exp_path, "outputs", "data")):
                    experiments = [exp_path]
        else:
            logger.error(f"Experiment path not found: {exp_path}")
            return
    else:
        experiments = find_all_experiments(results_root)

    if not experiments:
        logger.error("No experiments found!")
        return

    logger.info(f"Found {len(experiments)} experiment(s) to evaluate")
    logger.info("=" * 70)

    total_pass = 0
    total_fail = 0
    total_time = 0.0
    failed_experiments = []

    for exp_dir in experiments:
        rel_path = os.path.relpath(exp_dir, results_root)
        logger.info(f"\n{'='*70}")
        logger.info(f"Experiment: {rel_path}")
        logger.info(f"{'='*70}")

        try:
            result_str, ret_dict, elapsed = evaluate_experiment(
                exp_dir, args.data_root, logger, args.classes)
            total_time += elapsed
            logger.info(f"Evaluation time: {elapsed:.2f}s")

            if args.verify:
                log_path = os.path.join(exp_dir, "logs", "train.log")
                log_result = extract_eval_from_log(log_path)
                if log_result:
                    match, details = compare_results(result_str, log_result, args.tolerance)
                    if match:
                        logger.info(f"[PASS] Results match train.log")
                        total_pass += 1
                    else:
                        logger.warning(f"[FAIL] Results differ from train.log")
                        logger.warning(details)
                        total_fail += 1
                        failed_experiments.append(rel_path)
                else:
                    logger.warning(f"[SKIP] No eval results found in train.log")
        except Exception as e:
            logger.error(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            total_fail += 1
            failed_experiments.append(rel_path)

    logger.info(f"\n{'='*70}")
    logger.info(f"Summary: {len(experiments)} experiments evaluated in {total_time:.2f}s total")
    if args.verify:
        logger.info(f"  PASS: {total_pass}, FAIL: {total_fail}")
        if failed_experiments:
            logger.info(f"  Failed experiments:")
            for exp in failed_experiments:
                logger.info(f"    - {exp}")


if __name__ == "__main__":
    main()
