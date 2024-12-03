import json
import os

# Define paths
BASE_PATH = 'helper_folder/data/challenges/'
TRAINING_CHALLENGES_PATH = os.path.join(BASE_PATH, 'arc-agi_training_challenges.json')
TRAINING_SOLUTIONS_PATH = os.path.join(BASE_PATH, 'arc-agi_training_solutions.json')
OUTPUT_TRAIN_PATH = "training_data.jsonl"

# Template for fine-tuning messages
SYSTEM_MESSAGE = "You are an AI trained to solve grid-based reasoning problems."
USER_PROMPT_TEMPLATE = '''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
----------------------------------------
{training_data}
----------------------------------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
----------------------------------------
[{{'input': {input_test_data}, 'output': [[]]}}]
----------------------------------------
What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information.'''

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Preprocess a single task
def preprocess_task(file_name, grids, solutions):
    """
    Create a single fine-tuning example in the correct format.
    """
    train_grids = grids.get('train', [])
    test_inputs = grids.get('test', [])
    test_outputs = solutions.get(file_name, [])

    if not test_inputs or not test_outputs:
        return None  # Skip tasks with missing inputs or outputs

    # Format training examples
    training_data = "\n".join(
        f"Input: {example['input']}\nOutput: {example['output']}" for example in train_grids
    )

    # Extract the first test input and corresponding output
    input_test_data = test_inputs[0]['input']
    output_test_data = test_outputs[0]

    # Construct the message structure
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            training_data=training_data,
            input_test_data=input_test_data
        )},
        {"role": "assistant", "content": str(output_test_data)}
    ]

    return {"messages": messages}

# Preprocess the full dataset
def preprocess_data(challenges, solutions):
    """
    Preprocess all tasks in the dataset.
    """
    data = []
    for file_name, grids in challenges.items():
        processed = preprocess_task(file_name, grids, solutions)
        if processed:
            data.append(processed)
    return data

# Save to JSONL
def save_to_jsonl(data, output_path):
    """
    Save the dataset to a JSONL file.
    """
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def main():
    # Load datasets
    training_challenges = load_json(TRAINING_CHALLENGES_PATH)
    training_solutions = load_json(TRAINING_SOLUTIONS_PATH)

    # Preprocess datasets
    train_data = preprocess_data(training_challenges, training_solutions)

    # Save preprocessed data
    save_to_jsonl(train_data, OUTPUT_TRAIN_PATH)

    print(f"Training data saved to {OUTPUT_TRAIN_PATH} with {len(train_data)} examples.")

if __name__ == "__main__":
    main()
