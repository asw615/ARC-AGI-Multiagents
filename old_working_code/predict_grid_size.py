import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Set up color map
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
)
norm = colors.Normalize(vmin=0, vmax=10)

# Function to load data
def load_arc_data(challenge_path, solution_path):
    with open(challenge_path, 'rb') as f:
        challenges = json.load(f)
    with open(solution_path, 'rb') as f:
        solutions = json.load(f)
    return challenges, solutions

# Function to process inputs and outputs from challenges
def process_tasks(challenges, solutions):
    task_storage = []
    for key, task in challenges.items():
        train_inputs = [i['input'] for i in task['train']]
        train_outputs = [i['output'] for i in task['train']]
        test_inputs = [i['input'] for i in task['test']]
        test_outputs = solutions[key]
        task_storage.append([train_inputs, train_outputs, test_inputs, test_outputs])
    return task_storage

# Function to calculate grid sizes
def calculate_sizes(task_storage):
    size_data = []
    for train_inputs, train_outputs, test_inputs, test_outputs in task_storage:
        train_in_size = [[len(i[0]), len(i)] for i in train_inputs]
        train_out_size = [[len(i[0]), len(i)] for i in train_outputs]
        test_in_size = [[len(i[0]), len(i)] for i in test_inputs]
        test_out_size = [[len(i[0]), len(i)] for i in test_outputs]
        size_data.append([train_in_size, train_out_size, test_in_size, test_out_size])
    return size_data

# Function to predict grid sizes
def predict_grid_size(size_data):
    pred_sizes = []
    for train_in_size, train_out_size, test_in_size, test_out_size in size_data:
        ratio_x = train_in_size[0][0] / train_out_size[0][0]
        ratio_y = train_in_size[0][1] / train_out_size[0][1]

        # Check for consistent ratios
        for in_size, out_size in zip(train_in_size[1:], train_out_size[1:]):
            if in_size[0] / out_size[0] != ratio_x or in_size[1] / out_size[1] != ratio_y:
                break
        else:
            pred_test_size = [[int(i[0] / ratio_x), int(i[1] / ratio_y)] for i in test_in_size]
            pred_sizes.append(pred_test_size)
            continue
        
        # Check for consistent subtraction
        sub_x = train_in_size[0][0] - train_out_size[0][0]
        sub_y = train_in_size[0][1] - train_out_size[0][1]
        for in_size, out_size in zip(train_in_size[1:], train_out_size[1:]):
            if in_size[0] - out_size[0] != sub_x or in_size[1] - out_size[1] != sub_y:
                break
        else:
            pred_test_size = [[i[0] - sub_x, i[1] - sub_y] for i in test_in_size if i[0] - sub_x > 0 and i[1] - sub_y > 0]
            pred_sizes.append(pred_test_size)
            continue
        
        # Default last prediction if no pattern matches
        last_x, last_y = train_out_size[0][0], train_out_size[0][1]
        for in_size, out_size in zip(train_in_size[1:], train_out_size[1:]):
            if out_size[0] != last_x or out_size[1] != last_y:
                break
        else:
            pred_test_size = [[last_x, last_y] for _ in test_in_size]
            pred_sizes.append(pred_test_size)
            continue
        
        # Default to [-1, -1] if no prediction
        pred_test_size = [[-1, -1] for _ in test_in_size]
        pred_sizes.append(pred_test_size)
    
    return pred_sizes

# Function to evaluate prediction accuracy
def evaluate_predictions(pred_sizes, size_data):
    test_out_sizes = [i[3] for i in size_data]
    correct_preds = [i == j for i, j in zip(pred_sizes, test_out_sizes)]
    known_preds = [i[0] != [-1, -1] for i in pred_sizes]
    return np.mean(correct_preds), np.mean(known_preds), correct_preds, known_preds

# Visualization function
def visualize_task(task_storage, task_idx):
    graph_item = task_storage[task_idx]
    fig, ax = plt.subplots(2, len(graph_item[0]) + len(graph_item[2]), figsize=(15, 5))
    for i in range(len(graph_item[0])):
        ax[0][i].imshow(graph_item[0][i], cmap, norm)
    for i in range(len(graph_item[2])):
        ax[0][len(graph_item[0]) + i].imshow(graph_item[2][i], cmap, norm)
    for i in range(len(graph_item[1])):
        ax[1][i].imshow(graph_item[1][i], cmap, norm)
    plt.show()

# Main script execution
challenge_path = 'arc-agi-genesis/data/challenges/arc-agi_training_challenges.json'
solution_path = 'arc-agi-genesis/data/challenges/arc-agi_training_solutions.json'

# Load data
challenges, solutions = load_arc_data(challenge_path, solution_path)

# Process tasks
task_storage = process_tasks(challenges, solutions)

# Calculate sizes
size_data = calculate_sizes(task_storage)

# Predict grid sizes
pred_sizes = predict_grid_size(size_data)

# Evaluate predictions
mean_correct, mean_known, correct_preds, known_preds = evaluate_predictions(pred_sizes, size_data)
print(f'Accuracy: {mean_correct}, Known: {mean_known}')

# Visualize a task (Example: task index 194)
visualize_task(task_storage, 2)
print(f'Predicted sizes: {pred_sizes[2]}')