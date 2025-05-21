# OpenAlpha_Evolve: Contribute to Improve this Project

![openalpha_evolve_workflow](https://github.com/user-attachments/assets/9d4709ad-0072-44ae-bbb5-7eea1c5fa08c)

OpenAlpha_Evolve is an open-source Python framework inspired by the groundbreaking research on autonomous coding agents like DeepMind's AlphaEvolve. It's a **regeneration** of the core idea: an intelligent system that iteratively writes, tests, and improves code using Large Language Models (LLMs) via LiteLLM, guided by the principles of evolution.

Our mission is to provide an accessible, understandable, and extensible platform for researchers, developers, and enthusiasts to explore the fascinating intersection of AI, code generation, and automated problem-solving.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

## Table of Contents
- [✨ The Vision: AI-Driven Algorithmic Innovation](#-the-vision-ai-driven-algorithmic-innovation)
- [🧠 How It Works: The Evolutionary Cycle](#-how-it-works-the-evolutionary-cycle)
- [🚀 Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [🏁 Getting Started](#-getting-started)
- [💡 Defining Your Own Algorithmic Quests!](#-defining-your-own-algorithmic-quests)
- [🔮 The Horizon: Future Evolution](#-the-horizon-future-evolution)
- [🤝 Join the Evolution: Contributing](#-join-the-evolution-contributing)
- [📜 License](#-license)
- [🙏 Homage](#-homage)

---
![image](https://github.com/user-attachments/assets/ff498bb7-5608-46ca-9357-fd9b55b76800)
![image](https://github.com/user-attachments/assets/c1b4184a-f5d5-43fd-8f50-3e729c104e11)



## ✨ The Vision: AI-Driven Algorithmic Innovation

Imagine an agent that can:

*   Understand a complex problem description.
*   Generate initial algorithmic solutions.
*   Rigorously test its own code.
*   Learn from failures and successes.
*   Evolve increasingly sophisticated and efficient algorithms over time.

OpenAlpha_Evolve is a step towards this vision. It's not just about generating code; it's about creating a system that *discovers* and *refines* solutions autonomously.

---

## 🧠 How It Works: The Evolutionary Cycle

OpenAlpha_Evolve employs a modular, agent-based architecture to orchestrate an evolutionary process:

1.  **Task Definition**: You, the user, define the algorithmic "quest" – the problem to be solved, including examples of inputs and expected outputs.
2.  **Prompt Engineering (`PromptDesignerAgent`)**: This agent crafts intelligent prompts for the LLM. It designs:
    *   *Initial Prompts*: To generate the first set of candidate solutions.
    *   *Mutation Prompts*: To introduce variations and improvements to existing solutions, often requesting changes in a "diff" format.
    *   *Bug-Fix Prompts*: To guide the LLM in correcting errors from previous attempts, also typically expecting a "diff".
3.  **Code Generation (`CodeGeneratorAgent`)**: Powered by an LLM (currently configured for Gemini), this agent takes the prompts and generates Python code. If a "diff" is requested and received, it attempts to apply the changes to the parent code.
4.  **Evaluation (`EvaluatorAgent`)**: The generated code is put to the test!
    *   *Syntax Check*: Is the code valid Python?
    *   *Execution*: The code is run in a temporary, isolated environment against the input/output examples defined in the task.
    *   *Fitness Scoring*: Programs are scored based on correctness (how many test cases pass), efficiency (runtime), and other potential metrics.
5.  **Database (`DatabaseAgent`)**: All programs (code, fitness scores, generation, lineage) are stored, creating a record of the evolutionary history (currently in-memory).
6.  **Selection (`SelectionControllerAgent`)**: The "survival of the fittest" principle in action. This agent selects:
    *   *Parents*: Promising programs from the current generation to produce offspring.
    *   *Survivors*: The best programs from both the current population and new offspring to advance to the next generation.
7.  **Iteration**: This cycle repeats for a defined number of generations, with each new generation aiming to produce better solutions than the last.
8.  **Orchestration (`TaskManagerAgent`)**: The maestro of the operation, coordinating all other agents and managing the overall evolutionary loop.

---

## 🚀 Key Features

*   **LLM-Powered Code Generation**: Leverages state-of-the-art Large Language Models via LiteLLM, supporting multiple providers (OpenAI, Anthropic, Google, etc.).
*   **Evolutionary Algorithm Core**: Implements iterative improvement through selection, LLM-driven mutation/bug-fixing using diffs, and survival.
*   **Modular Agent Architecture**: Easily extend or replace individual components (e.g., use a different LLM, database, or evaluation strategy).
*   **Automated Program Evaluation**: Syntax checking and functional testing against user-provided examples with timeout mechanisms.
*   **Configuration Management**: Easily tweak parameters like population size, number of generations, LLM models, and API settings via `config/settings.py`.
*   **Detailed Logging**: Comprehensive logs provide insights into each step of the evolutionary process.
*   **Diff-based Mutations**: The system is designed to use diffs for mutations and bug fixes, allowing for more targeted code modifications by the LLM.
*   **Open Source & Extensible**: Built with Python, designed for experimentation and community contributions.

---

## 📂 Project Structure

```
./
├── agents/                  # Contains the core intelligent agents responsible for different parts of the evolutionary process. Each agent is in its own subdirectory.
│   ├── code_generator/      # Agent responsible for generating code using LLMs.
│   ├── database_agent/      # Agent for managing the storage and retrieval of programs and their metadata.
│   ├── evaluator_agent/     # Agent that evaluates the generated code for syntax, execution, and fitness.
│   ├── prompt_designer/     # Agent that crafts prompts for the LLM for initial generation, mutation, and bug fixing.
│   ├── selection_controller/  # Agent that implements the selection strategy for parent and survivor programs.
│   ├── task_manager/        # Agent that orchestrates the overall evolutionary loop and coordinates other agents.
│   ├── rl_finetuner/        # Placeholder for a future Reinforcement Learning Fine-Tuner agent to optimize prompts.
│   └── monitoring_agent/    # Placeholder for a future Monitoring Agent to track and visualize the process.
├── config/                  # Holds configuration files, primarily `settings.py` for system parameters and API keys.
├── core/                    # Defines core data structures and interfaces, like `Program` and `TaskDefinition`.
├── utils/                   # Contains utility functions and helper classes used across the project (currently minimal).
├── tests/                   # Includes unit and integration tests to ensure code quality and correctness (placeholders, to be expanded).
├── scripts/                 # Stores helper scripts for various tasks, such as generating diagrams or reports.
├── main.py                  # The main entry point to run the OpenAlpha_Evolve system and start an evolutionary run.
├── requirements.txt         # Lists all Python package dependencies required to run the project.
├── .env.example             # An example file showing the environment variables needed, such as API keys. Copy this to `.env` and fill in your values.
├── .gitignore               # Specifies intentionally untracked files that Git should ignore (e.g., `.env`, `__pycache__/`).
├── LICENSE.md               # Contains the full text of the MIT License under which the project is distributed.
└── README.md                # This file! Provides an overview of the project, setup instructions, and documentation.
```

---

## 🏁 Getting Started

1.  **Prerequisites**:
    *   Python 3.10+
    *   `pip` for package management
    *   `git` for cloning

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shyamsaktawat/OpenAlpha_Evolve.git
    cd OpenAlpha_Evolve
    ```

3.  **Set Up a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up Environment Variables (Crucial for API Keys)**:
    *   **This step is essential for the application to function correctly with your API keys.** The `.env` file stores your sensitive credentials and configuration, overriding the default placeholders in `config/settings.py`.
    *   Create your personal environment file by copying the example:
        ```bash
        cp .env.example .env
        ```


### LLM Configuration


6.  **Review Configuration (Optional)**:
    *   Open `config/settings.py`. While most LLM settings are now primarily managed via `.env` and LiteLLM, you can still review:
        *   Default model fallbacks if environment variables are not set (e.g., `LITELLM_DEFAULT_MODEL` has a fallback in `settings.py`).
        *   The specific model names used for evaluation (`EVALUATION_MODEL`) and fast operations (`FLASH_MODEL`) if not overridden in `.env`.
        *   Default LiteLLM parameters like `LITELLM_MAX_TOKENS`, `LITELLM_TEMPERATURE`, etc., which serve as defaults if not set in `.env`.
        *   Evolutionary parameters like `POPULATION_SIZE` and `GENERATIONS`.
        *   API retry settings or logging levels.

7.  **Run OpenAlpha_Evolve!**
    The `main.py` file is configured with an example task (Dijkstra's algorithm). To run it:
    ```bash
    python -m main
    ```
    Watch the logs in your terminal to see the evolutionary process unfold! Log files are also saved to `alpha_evolve.log` (by default).

8.  **Launch the Gradio Web Interface**
    Interact with the system via the web UI. To start the Gradio app:
    ```bash
    python app.py
    ```
    Gradio will display a local URL (e.g., http://127.0.0.1:7860) and a public share link if enabled. Open this in your browser to define custom tasks and run the evolution process interactively.

---

## 💡 Defining Your Own Algorithmic Quests!

Want to challenge OpenAlpha_Evolve with a new problem? It's easy:

1.  **Open `main.py`** (or your custom script where you define tasks).
2.  **Modify or create a `TaskDefinition` object**:
    *   `id`: A unique string identifier for your task (e.g., "sort_list_task", "find_max_with_bound_task").
    *   `description`: A clear, detailed natural language description of the problem. This is crucial for the LLM to understand what to do. Be specific about function names, expected behavior, and constraints.
    *   `function_name_to_evolve`: The name of the Python function the agent should try to create/evolve (e.g., "custom_sort", "find_max_above_threshold").
    *   `input_output_examples`: A list of dictionaries, each containing an `input` and the corresponding expected `output`. These are vital for evaluation.
        *   The `input` field for each example should be a list, where each element of the list corresponds to an argument for your function. If your function takes a single argument, this will be a list containing that one argument. For functions with multiple arguments, the list will contain all arguments in the correct order (e.g., `input=[[1, 2, 3], 0]` for a function `my_func(list_arg, threshold_arg)`).
        *   For numerical problems requiring positive or negative infinity, you can use `float('inf')` or `float('-inf')` directly in your Python code when defining these examples. The system's evaluation harness is designed to correctly serialize and deserialize these special float values. See the example below.
    *   `allowed_imports`: Specify a list of Python standard libraries that the generated code is allowed to import (e.g., `["heapq", "math", "sys"]`). This helps guide the LLM and can be important for the execution sandbox.
    *   (Optional) `evaluation_criteria`: Define how success is measured (currently primarily driven by correctness based on test cases).
    *   (Optional) `initial_code_prompt`: Override the default initial prompt if you need more specific instructions for the first code generation attempt.

3.  **Run the agent** with your new or modified task definition.

The quality of your `description` and the comprehensiveness of your `input_output_examples` significantly impact the agent's success!

### Example: TaskDefinition with `float('inf')`

Here's a conceptual example of how you might define a task that requires the use of `float('-inf')` as an initial comparison point for finding a maximum value within a certain range.

```python
from core.task_definition import TaskDefinition

find_max_task = TaskDefinition(
    id="find_max_in_list_with_lower_bound",
    description="Write a Python function called 'find_max_with_bound' that takes a list of numbers and a lower bound. "
                "The function should return the largest number in the list that is greater than or equal to the lower bound. "
                "If no such number exists, it should return float('-inf').",
    function_name_to_evolve="find_max_with_bound",
    input_output_examples=[
        {"input": [[1, 5, 2, 8, 3, 10], 3], "output": 10},
        {"input": [[-1, -5, -2, -8], 0], "output": float('-inf')},
        {"input": [[10, 20, 30], 10], "output": 30},
        {"input": [[5, 15, 25], 30], "output": float('-inf')},
        {"input": [[4, 8, 2], float('-inf')], "output": 8}, # Using -inf as a bound
        {"input": [[], 5], "output": float('-inf')}, # Empty list case
    ],
    allowed_imports=["math"] # Allow 'math' if needed, though float('-inf') is built-in
)

# To use this task, you would pass `find_max_task` to the TaskManagerAgent.
# Note: float('inf') and float('-inf') are used directly in the Python list.
# The system will handle their JSON serialization/deserialization during evaluation.
```

### Best Practices for Task Definition

Crafting effective task definitions is key to guiding OpenAlpha_Evolve successfully. Consider these tips:

*   **Be Clear and Unambiguous**: Write task descriptions as if you're explaining the problem to another developer. Avoid jargon where possible, or explain it clearly. The more precise your language, the better the LLM can interpret your intent.
*   **Provide Diverse and Comprehensive Examples**: Your `input_output_examples` are the primary way the agent verifies its generated code.
    *   Include typical use cases.
    *   Cover edge cases (e.g., empty lists, zero values, very large or small numbers, `None` inputs if applicable).
    *   Include examples that test different logical paths in the expected solution.
    *   Ensure the outputs are correct for each input.
*   **Start Simple, Then Increase Complexity**: If you have a complex problem, consider breaking it down or starting with a simpler version. Once the agent can solve the simpler task, you can gradually add more constraints or features to the description and examples. This iterative approach can be more effective than starting with a highly complex definition.
*   **Specify Constraints and Edge Cases in the Description**: Don't rely solely on examples to convey all requirements. If there are specific constraints (e.g., "the input list will always contain positive integers," "the function should handle lists up to 10,000 elements efficiently") or known edge cases that need special handling, mention them explicitly in the `description`.
*   **Define Expected Function Signature**: Clearly state the expected function name (`function_name_to_evolve`) and the nature of its parameters in the `description`. This helps the LLM generate code that matches your evaluation setup.
*   **Iterate and Refine**: Your first task definition might not be perfect. If the agent struggles or produces incorrect solutions, review your description and examples. Are they clear? Are there any ambiguities? Could more examples help? Iteratively refine your task definition based on the agent's performance.

---

## 🔮 The Horizon: Future Evolution

OpenAlpha_Evolve is a living project, and its future is shaped by community collaboration! Here are some directions we're excited to explore, with many opportunities for you to contribute:

*   **Advanced Evaluation Sandboxing**: Implementing more robust and secure sandboxing for code execution. This could involve using technologies like Docker containers, gVisor, or `nsjail` to better isolate code and manage complex dependencies, especially for tasks requiring external libraries. (Contribution Idea: Help integrate and test a new sandboxing technology!)
*   **Sophisticated Fitness Metrics**: Expanding beyond correctness and basic runtime. We aim to incorporate checks for code complexity (e.g., cyclomatic complexity using tools like `radon`), adherence to coding styles (e.g., via linters like Flake8 or Pylint), resource usage (memory profiling), and the ability to define custom domain-specific metrics for specialized tasks. (Contribution Idea: Propose and implement a new fitness metric or integrate a code analysis tool!)
*   **Reinforcement Learning for Prompt Strategy**: Activating and enhancing the `RLFineTunerAgent` to dynamically optimize prompt engineering strategies. This involves training a model that learns which prompt modifications lead to better code generation outcomes over time, potentially using feedback from fitness scores. (Contribution Idea: Design or implement RL algorithms for prompt optimization within the agent!)
*   **Enhanced Monitoring & Visualization**: Developing more comprehensive tools via the `MonitoringAgent` to visualize the evolutionary process. This could include interactive dashboards (perhaps using Streamlit or Plotly Dash) to track fitness landscapes, diversity of solutions, agent decision-making, and the lineage of successful programs. (Contribution Idea: Create new visualizations or integrate a dashboarding framework!)
*   **Expanded LLM Provider Support**: Continuously adding support for more LLM providers and models through LiteLLM's growing ecosystem. This ensures users have access to the latest and most suitable models for their tasks. (Contribution Idea: Add configuration and testing for a new LLM provider or a cutting-edge model!)
*   **Self-Correction & Reflection**: Enabling the agent to learn from its mistakes more deeply. This could involve mechanisms for analyzing execution errors, identifying patterns in failed test cases, or even prompting an LLM to reflect on why a particular solution was suboptimal and suggest improvements to its own problem-solving approach. (Contribution Idea: Develop new heuristics or LLM prompts for automated error analysis and reflection!)
*   **Diverse Task Domains**: Applying OpenAlpha_Evolve to a wider range of algorithmic problems. This includes exploring tasks in areas like scientific computing (e.g., physics simulations, bioinformatics), engineering design, financial modeling, data analysis pipelines, and even creative coding challenges. (Contribution Idea: Define a novel task in a new domain and share your results!)
*   **Community-Driven Task Library**: Building a rich, well-documented collection of interesting and challenging tasks contributed by the community. This library would serve as a benchmark and a source of inspiration for users. (Contribution Idea: Submit a well-defined task with examples to our upcoming task repository!)
*   **Improved Diff Application**: Making the diff application logic more robust and intelligent. This could involve better parsing of LLM-generated diffs, handling conflicts more gracefully, or exploring alternative structured formats for the LLM to suggest code modifications. (Contribution Idea: Research and implement more advanced diff parsing and application techniques!)
*   **Crossover Implementation**: Adding a classic genetic algorithm "crossover" (or recombination) mechanism. This would allow the system to combine parts of two successful parent programs to create offspring, offering an alternative or supplement to LLM-driven mutation for exploring the solution space. (Contribution Idea: Implement a crossover strategy suitable for code structures!)
*   **Multi-Objective Optimization**: Evolving solutions that balance multiple, potentially competing, fitness criteria. For instance, finding code that is not only correct but also fast, memory-efficient, and perhaps even easy to read. This would involve changes to the selection and fitness evaluation mechanisms. (Contribution Idea: Adapt the selection process to handle Pareto fronts or weighted-sum fitness for multiple objectives!)
*   **Integration with Formal Verification Tools**: For applications where correctness is absolutely critical (e.g., security-sensitive code, core infrastructure algorithms), exploring how generated code could be automatically fed into formal verification systems like TLA+, Coq, or Z3. This could provide stronger guarantees about the code's properties. (Contribution Idea: Investigate and prototype an interface to a formal verification tool for suitable tasks!)

We believe in the power of open source and community effort. If any of these areas spark your interest, please check out our contribution guidelines, open an issue to discuss your ideas, or submit a pull request!

---

## 🤝 Join the Evolution: Contributing

This is an open invitation to collaborate! Whether you're an AI researcher, a Python developer, or simply an enthusiast, your contributions are welcome.

*   **Report Bugs**: Find an issue? Please create an issue on GitHub!
*   **Suggest Features**: Have an idea to make OpenAlpha_Evolve better? Open an issue to discuss it!
*   **Submit Pull Requests**:
    *   Fork the repository.
    *   Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature-name`).
    *   Write clean, well-documented code.
    *   Add tests for your changes if applicable.
    *   Ensure your changes don't break existing functionality.
    *   Submit a pull request with a clear description of your changes!

Let's evolve this agent together!

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE.md` file for details.

---

## 🙏 Homage

OpenAlpha_Evolve is proudly inspired by the pioneering work of the Google DeepMind team on AlphaEvolve and other related research in LLM-driven code generation and automated discovery. This project aims to make the core concepts more accessible for broader experimentation and learning. We stand on the shoulders of giants.

---

*Disclaimer: This is an experimental project. Generated code may not always be optimal, correct, or secure. Always review and test code thoroughly, especially before using it in production environments.* 
