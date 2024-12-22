import json
import os
import csv
from typing import Tuple, Dict, Any, List
import statistics
import base64
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib import colors
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###############################################################################
# VISUALIZATION SETTINGS
###############################################################################
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF']
)
norm = colors.Normalize(vmin=0, vmax=10)

def openai_encode_image_base64(plt_obj):
    """
    Convert a Matplotlib figure into a base64 string.
    """
    import io
    buf = io.BytesIO()
    plt_obj.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    base64_image = base64.b64encode(buf.read()).decode('utf-8')
    plt_obj.close()
    return f"data:image/png;base64,{base64_image}"

def plot_eval(title_text: str,
              test_input: List[List[int]],
              predicted_solution: List[List[int]],
              actual_solution: List[List[int]]) -> str:
    """
    Plots the test input, predicted solution, and actual solution side-by-side, 
    then returns the figure as a base64 string for embedding in HTML.
    """
    try:
        test_input_arr = np.array(test_input)
        predicted_arr = np.array(predicted_solution)
        actual_arr = np.array(actual_solution)

        plt.close('all')
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title_text, fontsize=16, fontweight='bold')

        axs[0].imshow(test_input_arr, cmap=cmap, norm=norm)
        axs[0].set_title('Evaluation Input')
        axs[0].grid(True, color='lightgrey', linewidth=0.5)
        axs[0].set_xticks([x - 0.5 for x in range(1 + len(test_input_arr[0]))])
        axs[0].set_yticks([x - 0.5 for x in range(1 + len(test_input_arr))])

        axs[1].imshow(predicted_arr, cmap=cmap, norm=norm)
        axs[1].set_title('Model Prediction')
        axs[1].grid(True, color='lightgrey', linewidth=0.5)
        axs[1].set_xticks([x - 0.5 for x in range(1 + len(predicted_arr[0]))])
        axs[1].set_yticks([x - 0.5 for x in range(1 + len(predicted_arr))])

        axs[2].imshow(actual_arr, cmap=cmap, norm=norm)
        axs[2].set_title('Actual Solution')
        axs[2].grid(True, color='lightgrey', linewidth=0.5)
        axs[2].set_xticks([x - 0.5 for x in range(1 + len(actual_arr[0]))])
        axs[2].set_yticks([x - 0.5 for x in range(1 + len(actual_arr))])

        # Optionally annotate each cell with numeric values
        for ax, data in zip(axs, [test_input_arr, predicted_arr, actual_arr]):
            for i in range(len(data)):
                for j in range(len(data[0])):
                    ax.text(
                        j, i, str(data[i][j]),
                        ha='center',
                        va='center',
                        color='white' if data[i][j] in [0, 1, 9] else 'black'
                    )

        plt.tight_layout()
        return openai_encode_image_base64(plt)
    except Exception as e:
        logger.error(f"Error in plot_eval: {e}")
        plt.close('all')
        return ""

###############################################################################
# LOADING DATA
###############################################################################
def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

base_path = 'data/challenges/'
results_directory = r"C:\Users\Soren\Documents\ARC-AGI-Multiagents\results"

# Load subset of tasks (IDs) for 15x15
subset_eval_file = '28_15x15_evaluation.json'
subset_ids_evaluation = load_json(os.path.join(base_path, subset_eval_file))

# Load official evaluation solutions
evaluation_solutions = load_json(os.path.join(base_path, 'arc-agi_evaluation_solutions.json'))
evaluation_solutions = {k: v for k, v in evaluation_solutions.items() if k in subset_ids_evaluation}

# Load official evaluation challenges (to retrieve test inputs)
evaluation_challenges = load_json(os.path.join(base_path, 'arc-agi_evaluation_challenges.json'))
evaluation_challenges = {k: v for k, v in evaluation_challenges.items() if k in subset_ids_evaluation}

###############################################################################
# SCORING FUNCTIONS
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
    # Dimension check
    if len(attempt) != rows or any(len(row) != cols for row in attempt):
        return False, 0.0, 0

    total_pixels = rows * cols
    correct_pixels = 0
    for r in range(rows):
        for c in range(cols):
            if attempt[r][c] == solution[r][c]:
                correct_pixels += 1

    is_exact = (correct_pixels == total_pixels)
    percentage = (correct_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    return is_exact, percentage, correct_pixels

def score_submission(submission_file_name: str,
                     solutions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a submission file against provided solutions.
    Returns a dictionary with keys including:
      'submission_name', 'total_score', 'pixel_correctness_list', etc.
    """
    submission_name = os.path.basename(submission_file_name)
    print(f"Scoring {submission_name}")

    with open(submission_file_name, 'r', encoding='utf-8') as f:
        submission = json.load(f)

    # Filter to tasks that also exist in the solutions
    filtered_solutions = {task_id: solutions[task_id] for task_id in submission if task_id in solutions}

    total_score = 0.0
    total_tasks = 0
    pixel_correctness_list = []
    task_scores = {}
    task_pixel_details = {}

    # Iterate over each challenge in the submission
    for task_id, task_submission in submission.items():
        if task_id not in filtered_solutions:
            continue

        task_pairs = filtered_solutions[task_id]
        total_tasks += 1
        num_pairs = len(task_submission)
        task_score = 0.0
        task_pixel_details[task_id] = []

        # For each input-output pair
        for pair_index, pair_attempts in enumerate(task_submission):
            # We'll pick the best attempt
            pair_correct = False
            best_pixel_score = 0.0
            best_correct_count = 0

            for attempt_key, attempt_grid in pair_attempts.items():
                is_exact, pixel_percentage, correct_count = compare_solutions(attempt_grid, task_pairs[pair_index])
                if pixel_percentage > best_pixel_score:
                    best_pixel_score = pixel_percentage
                    best_correct_count = correct_count

                if is_exact:
                    pair_correct = True
                    break

            # If any attempt was exactly correct, score full credit for that pair
            if pair_correct:
                task_score += 1.0
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

        # Average correctness for this task (# correct pairs / total pairs)
        task_score /= num_pairs
        total_score += task_score
        task_scores[task_id] = task_score

    final_percentage = (total_score / total_tasks * 100) if total_tasks > 0 else 0
    results = {
        'submission_name': submission_name,
        'total_score': total_score,
        'total_tasks_scored': total_tasks,
        'percentage': round(final_percentage, 2),
        'pixel_correctness_list': pixel_correctness_list,
        'task_scores': task_scores,
        'task_pixel_details': task_pixel_details
    }
    return results

###############################################################################
# GATHER SUBMISSIONS & SCORE
###############################################################################
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
    full_path = os.path.join(results_directory, fname)
    if not os.path.exists(full_path):
        print(f"File {fname} not found in {results_directory}, skipping.")
        continue
    scored = score_submission(full_path, evaluation_solutions)
    all_results.append(scored)

###############################################################################
# WILCOXON TEST VS. REFERENCE (submission_4o.json)
###############################################################################
reference_name = "submission_4o.json"
reference_data = None
for item in all_results:
    if item['submission_name'] == reference_name:
        reference_data = item['pixel_correctness_list']
        break

wilcoxon_results = {}
if reference_data:
    for item in all_results:
        if item['submission_name'] != reference_name:
            if len(item['pixel_correctness_list']) == len(reference_data):
                w_stat, p_val = wilcoxon(item['pixel_correctness_list'], reference_data, zero_method='wilcox')
                wilcoxon_results[item['submission_name']] = (w_stat, p_val)
            else:
                wilcoxon_results[item['submission_name']] = (None, None)
else:
    logger.warning("Reference data not found or empty; skipping Wilcoxon tests.")

###############################################################################
# SAVE RESULTS TO CSV
###############################################################################
csv_path = os.path.join(results_directory, "evaluation_summary.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # Write header
    writer.writerow([
        "submission_name", 
        "total_score", 
        "total_tasks_scored", 
        "percentage", 
        "mean_pixel_correct", 
        "median_pixel_correct", 
        "wilcoxon_p_value"
    ])

    # Populate rows
    for item in all_results:
        mean_pc = statistics.mean(item['pixel_correctness_list']) if item['pixel_correctness_list'] else 0.0
        median_pc = statistics.median(item['pixel_correctness_list']) if item['pixel_correctness_list'] else 0.0

        # Wilcoxon p-value if available
        if item['submission_name'] in wilcoxon_results:
            _, p_val = wilcoxon_results[item['submission_name']]
            p_val_str = f"{p_val:.4f}" if p_val is not None else "N/A"
        elif item['submission_name'] == reference_name:
            p_val_str = "Reference"
        else:
            p_val_str = "N/A"

        writer.writerow([
            item['submission_name'],
            f"{item['total_score']:.2f}",
            item['total_tasks_scored'],
            item['percentage'],
            f"{mean_pc:.2f}",
            f"{median_pc:.2f}",
            p_val_str
        ])

print(f"CSV summary written to {csv_path}")

###############################################################################
# GENERATE HTML SUMMARY
###############################################################################
html_content = """
<!DOCTYPE html>
<html>
<head>
<title>ARC-AGI Evaluation Summary</title>
<style>
body {
  font-family: Arial, sans-serif;
  margin: 20px;
}
table {
  border-collapse: collapse;
  width: 90%;
  margin-bottom: 20px;
}
th, td {
  text-align: left;
  padding: 6px;
  border: 1px solid #ccc;
}
th {
  background-color: #f2f2f2;
}
.tab {
  overflow: hidden;
  border: 1px solid #ccc;
  background-color: #f1f1f1;
  margin-top: 40px;
}
.tab button {
  background-color: inherit;
  float: left;
  border: none;
  outline: none;
  cursor: pointer;
  padding: 14px 16px;
  transition: 0.3s;
  font-size: 16px;
}
.tab button:hover {
  background-color: #ddd;
}
.tab button.active {
  background-color: #ccc;
}
.tabcontent {
  display: none;
  padding: 20px 0px;
  border: 1px solid #ccc;
  border-top: none;
}
</style>
<script>
function openTab(evt, tabName) {
  // Hide all tabcontent
  var tabcontent = document.getElementsByClassName("tabcontent");
  for (var i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  // Remove "active" class from tablinks
  var tablinks = document.getElementsByTagName("button");
  for (var i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  // Show the current tab, and add "active" class
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}
</script>
</head>
<body>
<h1>ARC-AGI Evaluation Summary</h1>
<p>This report shows the performance of various submissions evaluated against the 28_15x15_evaluation set.</p>
"""

# Overall results table
html_content += "<h2>Overall Results</h2>"
html_content += "<table>"
html_content += "<tr><th>Submission</th><th>Total Score</th><th># Tasks</th><th>Percentage</th>"
html_content += "<th>Mean Pixel Correct</th><th>Median Pixel Correct</th><th>Wilcoxon p-value (vs. 4o)</th></tr>"

for item in all_results:
    mean_pc = statistics.mean(item['pixel_correctness_list']) if item['pixel_correctness_list'] else 0.0
    median_pc = statistics.median(item['pixel_correctness_list']) if item['pixel_correctness_list'] else 0.0

    # Wilcoxon p-value if available
    if item['submission_name'] in wilcoxon_results:
        _, p_val = wilcoxon_results[item['submission_name']]
        if p_val is not None:
            p_val_str = f"{p_val:.4f}"
        else:
            p_val_str = "N/A"
    elif item['submission_name'] == reference_name:
        p_val_str = "Reference"
    else:
        p_val_str = "N/A"

    html_content += (
        f"<tr>"
        f"<td>{item['submission_name']}</td>"
        f"<td>{item['total_score']:.2f}</td>"
        f"<td>{item['total_tasks_scored']}</td>"
        f"<td>{item['percentage']}%</td>"
        f"<td>{mean_pc:.2f}%</td>"
        f"<td>{median_pc:.2f}%</td>"
        f"<td>{p_val_str}</td>"
        f"</tr>"
    )
html_content += "</table>"

# Create tabs for each submission
html_content += '<div class="tab">'
for item in all_results:
    sub_name_noext = os.path.splitext(item['submission_name'])[0]
    html_content += f'<button class="tablinks" onclick="openTab(event, \'{sub_name_noext}\')">{item["submission_name"]}</button>'
html_content += '</div>'

# Detailed per-task sections
for item in all_results:
    sub_name_noext = os.path.splitext(item['submission_name'])[0]
    html_content += f'<div id="{sub_name_noext}" class="tabcontent">'
    html_content += f'<h2>Submission: {item["submission_name"]}</h2>'

    # Attempt to load the same JSON submission again to retrieve predictions
    submission_path = os.path.join(results_directory, item['submission_name'])
    if os.path.exists(submission_path):
        with open(submission_path, 'r', encoding='utf-8') as f:
            submission_json = json.load(f)

        # For each task
        for task_id, solution_pairs in submission_json.items():
            if task_id not in evaluation_solutions:
                continue

            html_content += f"<h3>Task: {task_id}</h3>"
            html_content += "<table>"
            html_content += "<tr><th>Pair Index</th><th>Correct Pixels</th><th>Pixel %</th><th>Visualization</th></tr>"

            # Retrieve stored pixel details
            if task_id not in item['task_pixel_details']:
                html_content += "</table>"
                continue

            # Each test pair
            for pair_index, pair_attempts in enumerate(solution_pairs):
                # match up details
                details_for_pair = [
                    d for d in item['task_pixel_details'][task_id]
                    if d["pair_index"] == pair_index
                ]
                if not details_for_pair:
                    continue

                correct_pixels = details_for_pair[0]["correct_pixels"]
                pixel_percentage = details_for_pair[0]["pixel_percentage"]

                # We'll pick the first attempt for visualization
                first_key = list(pair_attempts.keys())[0]
                predicted_solution = pair_attempts[first_key]

                # Attempt to load the actual test input
                try:
                    test_input_data = evaluation_challenges[task_id]['test'][pair_index]['input']
                    # Ground truth
                    ground_truth = evaluation_solutions[task_id][pair_index]
                    encoded_plot = plot_eval(
                        f"Task {task_id}, Pair {pair_index}",
                        test_input_data,
                        predicted_solution,
                        ground_truth
                    )
                except Exception as e:
                    logger.error(f"Error loading test input for {task_id}, pair {pair_index}: {e}")
                    encoded_plot = ""

                html_content += (
                    "<tr>"
                    f"<td>{pair_index}</td>"
                    f"<td>{correct_pixels}</td>"
                    f"<td>{pixel_percentage:.2f}%</td>"
                    f"<td><img src='{encoded_plot}' style='max-width:400px; border:1px solid #ccc;'/></td>"
                    "</tr>"
                )
            html_content += "</table>"
    else:
        html_content += "<p>Submission file not found; cannot generate detailed plots.</p>"
    html_content += "</div>"

html_content += """
<script>
// Open the first tab by default
document.getElementsByClassName('tablinks')[0].click();
</script>
</body>
</html>
"""

html_output_path = os.path.join(results_directory, "evaluation_summary.html")
with open(html_output_path, 'w', encoding='utf-8') as out_html:
    out_html.write(html_content)

print(f"HTML summary written to {html_output_path}")
