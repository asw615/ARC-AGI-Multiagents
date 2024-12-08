# Enhancing Reasoning Capabilities of Large Language Models Using Agents

## Overview

This project investigates how integrating agents can enhance the reasoning capabilities of Large Language Models (LLMs) for solving complex grid-based tasks. Utilizing the [LangChain](https://langchain.com/) framework, the system orchestrates multiple LLMs, using a Mixture of Agents (MoA) with LLMs from OpenAI and Anthropic, to analyze, generate, and execute solutions. The objective is to improve task-solving accuracy and efficiency by decomposing problems and managing workflows through a state graph. Using finetuning techniques have also been explored, though the main focus has been exploring multi-agent systems. 

**About ARC-AGI**

The **Abstraction and Reasoning Corpus for Artificial General Intelligence [ARC-AGI](https://arcprize.org/)** is a benchmark introduced by François Chollet in 2019 to evaluate an AI system's ability to generalize and acquire new skills beyond its training data. It consists of grid-based tasks where each cell can be one of ten colors, challenging models to identify patterns and generate correct outputs based on minimal examples. ARC-AGI emphasizes general intelligence by testing skill acquisition efficiency rather than task-specific proficiency, making it a pivotal measure in the pursuit of artificial general intelligence. :contentReference[oaicite:0]{index=0}


## Features

- **Agent-Based Architecture**: Breaks down and solves grid manipulation tasks using specialized agents.
- **Multi-LLM Integration**: Incorporates OpenAI's GPT-4o and Anthropics Claude sonnet 3.5.
- **State Graph Management**: Organizes the workflow of agents using a state graph for structured reasoning.
- **Tool utilization of agents**: A set of functions are defined for grid manipulation which the agents can use.
- **Data Handling**: Loads and processes training, evaluation, and test datasets from JSON files.
- **Visualization**: Generates plots to visualize input grids, model predictions, and actual solutions.
- **Fine-Tuning Capability**: Supports fine-tuning of LLMs to specialize in grid-based reasoning tasks.

## The two main scripts:
1. **'agents_w_tools.ipynb'**: This is the primary jupyter notebook which utilizes the agentic workflow to generate predicitons for the tasks. The notebook can be run either using API keys stored in a file called 'api.env' or locally using [Ollama](https://ollama.com/) framework. The notebook loads in the json files, generate tools for the agents to, sets up the agentic workflow and generates predictions with plots.
2. **single_agent.ipynb**: This jupyter notebook can be used to run a single agent. The notebook can be used to test general models or finetuned models. The notebook is a bit simpler, and does not include plotting when predicting.  

## Agentic Framework
The model processes each task in a linear agentic framework by:

**Identifying Logic:** Analyzing task descriptions and patterns.
**Recognizing Patterns:** Selecting relevant tools for the task.
**Generating Code:** Creating a solve_task function to manipulate grids.
**Revising Code:** Refining the generated code for accuracy and efficiency.
**Executing Code:** Running the solve_task function to generate predictions.

## Configuration and Usage
1. **Clone the Repository**: 
git clone https://github.com/asw615/ARC-AGI-Multiagents.git
2. **Create a Virtual Environment**: 
python3 -m venv venv
source venv/bin/activate
3. **Install Dependencies**: 
pip install -r requirements.txt
4. **Configure API Keys**: 
Create a file called 'api.env'. This file should include your API keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
5. **Running the notebooks**: 
The notebooks should now be runnable and ready for testing. Use 'agents_w_tools.ipynb' and specify which model you want to run on the task. By default, the models are run on 28 selected tasks from the evaluation set which all contain sets with grids smaller than 15x15.

## Viewing Results

After running the model, results are saved in the following files:

- **`submission_agents.json`**: Contains the predictions for each task.
- **`detailed_outputs.json`**: Provides detailed information about each prediction attempt.
- **`rewoolanggraph.log`**: Logs detailed execution information for debugging.

## Future Steps

To further enhance the system, future steps include:

1. **Enhanced Agentic Framework Using Hierarchical Agent Teams**:
   - Divide agents into specialized teams to handle distinct subtasks:
     - **Grid Size Prediction Team**: Focuses on determining the dimensions of the output grid.
     - **Pixel Placement Team**: Identifies where pixels should be placed within the grid.
     - **Color Prediction Team**: Determines the correct colors for the pixels.
   - These teams will operate under a hierarchical framework, where higher-level agents oversee task distribution and ensure coherence between subtasks.
   - A 'Supervisor' agent should be implemented to determine whether the solution matches the logic found.

2. **Enhancing Usage of Tools**: 
    - Currently the agents which are in charge of using tools are prompted with all of the tools. This framework could be improved by using the tool framework from Langchain. 

3. **Enhanced Prompts**
    - A wide testing of different prompts to see how much this increases prediction rates. Documentation from prompt engineering could be used to enhance the prompts. 

4. **Integration of RAG system** 
    - A RAG system for the agents could be used with written out explanations of how to manipulate the grids as well as examples of how other tasks was solved. For the current framework the agents have to learn a new set of skill for each task, whereas having a database of examples from how the tasks could be solved could potentially improve perfomance. 