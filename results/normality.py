import os
import json
import statistics
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from typing import Tuple, Dict, Any, List
from matplotlib import colors
from scipy.stats import ttest_ind, shapiro, kstest
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# CONFIGURATION
###############################################################################
# Adjust these paths as needed
BASE_PATH = 'data/challenges/'
RESULTS_DIRECTORY = r"C:\Users\Soren\Documents\ARC-AGI-Multiagents\results"
EVAL_FILE_15x15 = '28_15x15_evaluation.json'     # Subset of evaluation tasks
EVAL_SOLUTIONS_FILE = 'arc-agi_evaluation_solutions.json'
REFERENCE_SUBMISSION = "submission_4o.json"     # The model to compare against

###############################################################################
# STEP 1: LOAD EVALUATION DATA
###############################################################################
def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

base_eval_path = os.path.join(BASE_PATH, EVAL_FILE_15x15)
subset_ids_evaluation = load_json(base_eval_path)

eval_solutions_path = os.path.join(BASE_PATH, EVAL_SOLUTIONS_FILE)
evaluation_solutions = load_json(eval_solutions_path)
evaluation_solutions = {
    k: v for k, v in evaluation_solutions.items() if k in subset_ids_evaluation
}

###############################################################################
# STEP 2: SCORING LOGIC
###############################################################################
def compare_solutions(attempt: List[List[int]], solution: List[List[int]]) -> Tuple[bool, float, int]:
    """
    Compare an attempt grid against the correct solution grid.
    Returns (is_exact_match, percentage_correct_pixels, correct_pixel_count).
    """
    rows = len(solution)
    if rows == 0:
        return False, 0.0, 0
    
    cols = len(solution[0])
    # Check dimension mismatch
    if len(attempt) != rows or any(len(row) != cols for row in attempt):
        return False, 0.0, 0
    
    total_pixels = rows * cols
    correct_pixels = 0
    for r in range(rows):
        for c in range(cols):
            if attempt[r][c] == solution[r][c]:
                correct_pixels += 1
    
    is_exact = (correct_pixels == total_pixels)
    percentage = (correct_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
    return is_exact, percentage, correct_pixels

def score_submission(submission_file_name: str, solutions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a submission file against provided solutions.
    Returns a dictionary with:
    {
      'submission_name': str,
      'total_score': float,
      'total_tasks_scored': int,
      'percentage': float,
      'pixel_correctness_list': List[float],
      'task_scores': Dict[str, float],
      'task_pixel_details': Dict[str, List[Dict]]]  # additional detail for each task/pair
    }
    """
    submission_name = os.path.basename(submission_file_name)
    print(f"Scoring {submission_name}")

    # Load the submission
    with open(submission_file_name, "r", encoding="utf-8") as file:
        submission = json.load(file)

    # Filter solutions to only those present in submission
    filtered_solutions = {task_id: solutions[task_id] for task_id in submission if task_id in solutions}

    total_score = 0.0
    total_tasks = 0
    pixel_correctness_list = []
    task_scores = {}
    task_pixel_details = {}

    for task_id, task_submission in submission.items():
        if task_id not in filtered_solutions:
            continue

        task_pairs = filtered_solutions[task_id]
        total_tasks += 1
        task_score = 0.0
        num_pairs = len(task_submission)
        task_pixel_details[task_id] = []

        for pair_index, pair_attempts in enumerate(task_submission):
            # best_pixel_score is the highest pixel correctness among attempts
            best_pixel_score = 0.0
            best_correct_count = 0
            pair_correct = False
            solution = task_pairs[pair_index]  # ground truth solution for this pair

            for attempt_key, attempt_grid in pair_attempts.items():
                is_exact, pixel_percentage, correct_count = compare_solutions(attempt_grid, solution)

                if pixel_percentage > best_pixel_score:
                    best_pixel_score = pixel_percentage
                    best_correct_count = correct_count

                if is_exact:
                    pair_correct = True
                    break

            if pair_correct:
                task_score += 1
                pixel_correctness_list.append(100.0)
                task_pixel_details[task_id].append({
                    "pair_index": pair_index,
                    "correct_pixels": best_correct_count,
                    "pixel_percentage": 100.0
                })
            else:
                pixel_correctness_list.append(best_pixel_score)
                task_pixel_details[task_id].append({
                    "pair_index": pair_index,
                    "correct_pixels": best_correct_count,
                    "pixel_percentage": best_pixel_score
                })

        task_score /= num_pairs
        total_score += task_score
        task_scores[task_id] = task_score

    final_percentage = (total_score / total_tasks * 100) if total_tasks > 0 else 0.0
    return {
        'submission_name': submission_name,
        'total_score': total_score,
        'total_tasks_scored': total_tasks,
        'percentage': round(final_percentage, 2),
        'pixel_correctness_list': pixel_correctness_list,
        'task_scores': task_scores,
        'task_pixel_details': task_pixel_details
    }

###############################################################################
# STEP 3: GATHER RESULTS FROM THE RESULTS FOLDER
###############################################################################
def gather_all_results(results_dir: str, solutions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Iterates over a predefined list of submission files or all JSON files in
    the results directory, scores them, and returns the aggregated results.
    """
    # You may adjust this if you want to auto-discover all JSON files instead:
    # submission_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    submission_files = [
        "detailed_outputs.json",
        "submission_4o.json",
        "submission_4omini.json",
        "submission_agents.json",
        "submission_agentswtool.json",
        "submission_finetune4o.json",
        "submission_finetune4omini.json"
    ]

    all_results = []
    for fname in submission_files:
        full_path = os.path.join(results_dir, fname)
        if not os.path.exists(full_path):
            print(f"File {fname} not found in {results_dir}, skipping.")
            continue
        result = score_submission(full_path, solutions)
        all_results.append(result)
    return all_results

###############################################################################
# STEP 4: NORMALITY CHECKS
###############################################################################
def compute_paired_differences(
    reference_scores: List[float],
    other_scores: List[float]
) -> np.ndarray:
    """
    Computes the difference array (other - reference) on a per-task basis.
    Expects both lists to be of the same length and correspond to the same tasks.
    """
    if len(reference_scores) != len(other_scores):
        raise ValueError("Reference and other scores must have the same length for paired comparison.")
    differences = np.array([o - r for o, r in zip(other_scores, reference_scores)])
    return differences

def perform_normality_tests(differences: np.ndarray) -> Tuple[float, float]:
    """
    Performs Shapiro–Wilk and Kolmogorov–Smirnov tests on the distribution
    of differences and returns the corresponding p-values as a tuple.
    """
    # Shapiro–Wilk
    shapiro_stat, shapiro_p = shapiro(differences)
    # K–S
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    ks_stat, ks_p = kstest(differences, 'norm', args=(mean_diff, std_diff))
    return shapiro_p, ks_p

def plot_histogram_and_qq(differences: np.ndarray, model_name: str, reference_name: str) -> None:
    """
    Plots a histogram + KDE of the differences and a Q–Q plot in a single figure.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram with KDE
    sns.histplot(differences, kde=True, ax=axs[0], color='steelblue')
    axs[0].set_title(f"Histogram of Differences\n({model_name} - {reference_name})")
    axs[0].set_xlabel("Pixel Correctness Difference")
    axs[0].set_ylabel("Frequency")

    # Right: Q–Q Plot
    stats.probplot(differences, plot=axs[1])
    axs[1].set_title(f"Q–Q Plot of Differences\n({model_name} - {reference_name})")

    plt.tight_layout()
    plt.show()

def check_normality_for_all_submissions(all_results: List[Dict[str, Any]], reference_name: str):
    """
    Finds the reference submission in all_results, then for each other submission,
    computes (other - reference) differences in pixel correctness, runs normality
    tests, and plots the distributions.
    """
    reference_scores = None
    for item in all_results:
        if item["submission_name"] == reference_name:
            reference_scores = item["pixel_correctness_list"]
            break

    if reference_scores is None:
        print(f"Reference submission '{reference_name}' not found or no data.")
        return

    for item in all_results:
        if item["submission_name"] == reference_name:
            continue  # Skip comparing the reference to itself
        model_scores = item["pixel_correctness_list"]
        if len(model_scores) != len(reference_scores):
            print(f"Skipping {item['submission_name']}: length mismatch with reference.")
            continue

        differences = compute_paired_differences(reference_scores, model_scores)
        shapiro_p, ks_p = perform_normality_tests(differences)

        mean_diff = np.mean(differences)
        median_diff = np.median(differences)

        print("==============================================================")
        print(f"Normality Check for {item['submission_name']} vs. {reference_name}")
        print(f"Number of tasks: {len(differences)}")
        print(f"Mean difference:   {mean_diff:.2f}")
        print(f"Median difference: {median_diff:.2f}")
        print(f"Shapiro–Wilk p-value: {shapiro_p:.4f} (p < 0.05 => non-normal)")
        print(f"K–S p-value:         {ks_p:.4f} (p < 0.05 => non-normal)")

        plot_histogram_and_qq(differences, item['submission_name'], reference_name)

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    # 1) Gather all results by scoring each submission in the results directory
    all_results = gather_all_results(RESULTS_DIRECTORY, evaluation_solutions)
    
    # 2) Perform normality checks on each submission compared to the reference
    check_normality_for_all_submissions(all_results, REFERENCE_SUBMISSION)
