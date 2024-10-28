## Importing necessary libraries and dependencies
import os
import json
from langchain import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from langchain_core.pydantic_v1 import Field
from typing import List

# Loading environment variables (API keys)
from dotenv import load_dotenv
load_dotenv('api.env')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Setting up OpenAI LLM
llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key=openai_api_key, max_tokens=3000)

# Function to load JSON files
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Load the task data
base_path = 'arc-agi-genesis/data/challenges/'
training_challenges = load_json(base_path + 'arc-agi_training_challenges.json')
training_solutions = load_json(base_path + 'arc-agi_training_solutions.json')

task_sets = {
    'training': {
        'challenges': training_challenges,
        'solutions': training_solutions,
    }
}

# Defining the structure of the expected LLM response
class ARCPrediction(BaseModel):
    prediction: List[List] = Field(..., description="A prediction for a task")

# Function to convert JSON task into string format
def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    json_task = challenge_tasks[task_id]
    final_output = "Training Examples\n"
    
    train_tasks = json_task['train']
    test_task = json_task['test']

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n[" + "\n".join(str(row) for row in task['input']) + "]\n"
        final_output += f"Example {i + 1}: Output\n[" + "\n".join(str(row) for row in task['output']) + "]\n\n"
    
    final_output += "Test\n[" + "\n".join(str(row) for row in test_task[test_input_index]['input']) + "]\n\nYour Response:"
    return final_output

# Create a prompt template and parser
def create_prediction_tool():
    parser = JsonOutputParser(pydantic_object=ARCPrediction)

    # Define the prompt template
    prompt = PromptTemplate(
        template="You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern." 
                 "Identify the pattern, then apply that pattern to the test input to give a final output."
                 "Just give valid json list of lists response back, nothing else. Do not explain your thoughts."
                 "{format_instructions}\n{task_string}\n",
        input_variables=["task_string"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Create a chain to execute the prompt with the LLM and parser
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    def predict_task(input_data):
        task_string = input_data["task_string"]
        return llm_chain.run({"task_string": task_string})

    # Define the tool to be used by the agent
    return Tool(name="PredictionTool", func=predict_task, description="Solves ARC tasks by predicting the output based on the input.")

# Create a simple agent executor
def setup_agent():
    prediction_tool = create_prediction_tool()
    tools = [prediction_tool]  # Add more tools here in the future
    agent = AgentExecutor(agent_tools=tools, agent_prompt="Solve the task using the prediction tool.")
    return agent

# Main function to handle task predictions
def get_task_prediction(agent, challenge_tasks, task_id, test_input_index) -> List[List]:
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)

    # Run the agent to get the prediction
    result = agent.run({"task_string": task_string})
    
    # Parse the result
    parsed_result = ARCPrediction.parse_raw(result)
    
    # Optional sanity check on output type
    if not all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in parsed_result.prediction):
        raise ValueError("Output must be a list of lists of integers.")

    return parsed_result.prediction

# Running a prediction example
if __name__ == "__main__":
    agent = setup_agent()
    challenges, _ = load_tasks_from_file(task_set=task_sets['training'])
    prediction = get_task_prediction(agent, challenges, '0520fde7', 0)
    print(prediction)
