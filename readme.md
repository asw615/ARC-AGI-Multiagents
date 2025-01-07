# 🚀 Enhancing Reasoning Capabilities of Large Language Models Using Agents

## Overview

This project investigates how integrating agents can enhance the reasoning capabilities of Large Language Models (LLMs) for solving complex grid-based tasks. Utilizing the [LangChain](https://langchain.com/) framework, the system orchestrates multiple LLMs, using a Mixture of Agents (MoA) with LLMs from OpenAI and Anthropic, to analyze, generate, and execute solutions. The objective is to improve task-solving accuracy and efficiency by decomposing problems and managing workflows through a state graph. Using finetuning techniques have also been explored, though the main focus has been exploring multi-agent systems. 

**🌟 About ARC-AGI**

The **Abstraction and Reasoning Corpus for Artificial General Intelligence [ARC-AGI](https://arcprize.org/)** is a benchmark introduced by François Chollet in 2019 to evaluate an AI system's ability to generalize and acquire new skills beyond its training data. It consists of grid-based tasks where each cell can be one of ten colors, challenging models to identify patterns and generate correct outputs based on minimal examples. ARC-AGI emphasizes general intelligence by testing skill acquisition efficiency rather than task-specific proficiency, making it a pivotal measure in the pursuit of artificial general intelligence. 

![ARC-AGI Example Task](https://arcprize.org/media/images/arc-example-task.jpg)

## 🛠️ Features

- **Agent-Based Architecture**: Breaks down and solves grid manipulation tasks using specialized agents.
- **Multi-LLM Integration**: Incorporates OpenAI's GPT-4o and Anthropics Claude sonnet 3.5.
- **State Graph Management**: Organizes the workflow of agents using a state graph for structured reasoning.
- **Tool utilization of agents**: A set of functions are defined for grid manipulation which the agents can use.
- **Data Handling**: Loads and processes training, evaluation, and test datasets from JSON files.
- **Visualization**: Generates plots to visualize input grids, model predictions, and actual solutions.
- **Fine-Tuning Capability**: Supports fine-tuning of LLMs to specialize in grid-based reasoning tasks.

## 📂 The Main Scripts:
1. **'agents_w_tools.ipynb'**: This is the primary jupyter notebook which utilizes the agentic workflow to generate predicitons for the tasks. The notebook can be run either using API keys stored in a file called 'api.env' or locally using [Ollama](https://ollama.com/) framework. The notebook loads in the json files, generate tools for the agents to, sets up the agentic workflow and generates predictions with plots.
2. **'single_agent.ipynb'**: This jupyter notebook can be used to run a single agent. The notebook can be used to test general models or finetuned models. The notebook is a bit simpler, and does not include plotting when predicting.  

## 🤖 Agentic Framework
The model processes each task in a linear agentic framework by:

**Identifying Logic:** Analyzing task descriptions and patterns.<br>
**Recognizing Patterns:** Selecting relevant tools for the task.<br>
**Generating Code:** Creating a solve_task function to manipulate grids.<br>
**Revising Code:** Refining the generated code for accuracy and efficiency.<br>
**Executing Code:** Running the solve_task function to generate predictions.<br>

## 📝 Configuration and Usage
1. **Clone the Repository**:<br> 
git clone https://github.com/asw615/ARC-AGI-Multiagents.git
2. **Create a Virtual Environment**: <br>
python3 -m venv venv <br>
source venv/bin/activate
3. **Install Dependencies**: <br>
pip install -r requirements.txt
4. **Configure API Keys**: <br>
Create a file called 'api.env'. This file should include your API keys. <br>
OPENAI_API_KEY=your_openai_api_key_here <br>
ANTHROPIC_API_KEY=your_anthropic_api_key_here
5. **Running the notebooks**: <br>
The notebooks should now be runnable and ready for testing. Use 'agents_w_tools.ipynb' and specify which model you want to run on the task. By default, the models are run on 28 selected tasks from the evaluation set which all contain sets with grids smaller than 15x15.

## 🔍 Viewing Results

After running the model, results are saved in the following files:

- **`submission_agents.json`**: Contains the predictions for each task.
- **`detailed_outputs.json`**: Provides detailed information about each prediction attempt.
- **`rewoolanggraph.log`**: Logs detailed execution information for debugging.

Unfortunately, the performance of the agentic framework did not surpass baseline results. <br>
Fine-tuning - especially at test-time - seems to be a very promising strategy as shown from the 2024 [ARC Prize](https://arcprize.org/2024-results) competition <br><br>
The results can be seen in the table below. <br>
![Results]([https://arcprize.org/media/images/arc-example-task.jpg](https://github.com/asw615/ARC-AGI-Multiagents/blob/main/results/table_of_results.png))

## 📈 Future Steps

To further enhance the system, future steps include:

1. **Enhanced Agentic Framework Using Hierarchical Agent Teams**:
   - Divide agents into specialized teams to handle distinct subtasks:
     - **Grid Size Prediction Team**: Focuses on determining the dimensions of the output grid.
     - **Pixel Placement Team**: Identifies where pixels should be placed within the grid.
     - **Color Prediction Team**: Determines the correct colors for the pixels.
   - These teams will operate under a hierarchical framework, where higher-level agents oversee task distribution and ensure coherence between subtasks.
   - A 'Supervisor' agent should be implemented to determine whether the solution matches the logic found.

2. **Tool Optimization**:
   - Refine the tool usage framework by leveraging LangChain’s integrated tool system.
   - Enable agents to dynamically query and select only relevant tools based on task context, reducing unnecessary computations.

3. **Prompt Optimization**:
   - Systematically evaluate and refine prompts using principles from prompt engineering.
   - Incorporate clear examples and detailed instructions to improve prediction rates.

4. **Knowledge Integration via Retrieval-Augmented Generation (RAG)**:
   - Use a RAG system to provide agents with a database of solved task examples and explanations.
   - Enable agents to leverage prior knowledge to generalize solutions for unseen tasks, improving skill-acquisition efficiency.
