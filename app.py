import os
import sys
import json
import asyncio
import logging
import ast
import gradio as gr
from typing import Dict, Any, List, Optional

# Ensure project root is in path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition, Program
from config import settings

# Configure logging for the web interface
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Global variables to store current task and results
current_task = None
evolution_results = []
task_running = False

def validate_json(json_str):
    """Validate if a string is valid JSON."""
    try:
        json.loads(json_str)
        return True
    except:
        return False
    
def validate_examples(examples_str):
    """Validate if input/output examples are correctly formatted."""
    try:
        examples = json.loads(examples_str)
        if not isinstance(examples, list):
            return False, "Examples must be a JSON array of objects."
        
        for ex in examples:
            if "input" not in ex or "output" not in ex:
                return False, "Each example must have 'input' and 'output' keys."
            
        return True, ""
    except Exception as e:
        return False, f"Invalid JSON format: {str(e)}"
    
def parse_float_values(obj):
    """
    Recursively traverse a JSON-like object and replace string representations
    of infinity with actual float('inf') values.
    """
    if isinstance(obj, dict):
        return {k: parse_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_float_values(item) for item in obj]
    elif isinstance(obj, str):
        if obj == "Infinity" or obj == "float('inf')":
            return float('inf')
        elif obj == "-Infinity" or obj == "float('-inf')":
            return float('-inf')
        elif obj == "NaN" or obj == "float('nan')":
            return float('nan')
        return obj
    else:
        return obj

async def create_task_definition(
    task_id: str,
    description: str,
    function_name: str,
    examples_str: str,
    imports_str: str
) -> Optional[TaskDefinition]:
    """Create a TaskDefinition object from user inputs."""
    
    # Validate inputs
    if not task_id or not description or not function_name or not examples_str:
        return None, "All fields except allowed imports are required."
    
    # Validate examples
    is_valid, error_msg = validate_examples(examples_str)
    if not is_valid:
        return None, error_msg
    
    # Parse examples and handle float('inf') values
    examples = json.loads(examples_str)
    examples = parse_float_values(examples)
    
    # Parse allowed imports
    allowed_imports = []
    if imports_str:
        allowed_imports = [imp.strip() for imp in imports_str.split(',')]
    
    task = TaskDefinition(
        id=task_id,
        description=description,
        function_name_to_evolve=function_name,
        input_output_examples=examples,
        allowed_imports=allowed_imports
    )
    
    return task, "Task definition created successfully."

async def run_evolution(task: TaskDefinition, status_callback=None):
    """Run the evolutionary process on the defined task."""
    global task_running, evolution_results
    
    if task_running:
        return "Task is already running. Please wait."
    
    task_running = True
    evolution_results = []
    
    try:
        task_manager = TaskManagerAgent(task_definition=task)
        
        if status_callback:
            status_callback("Initializing population...")
        
        # Execute the evolutionary process
        best_programs = await task_manager.execute()
        
        if best_programs:
            logger.info(f"Evolution completed. Found {len(best_programs)} programs.")
            evolution_results = best_programs
            return f"Evolution completed. Found {len(best_programs)} solutions."
        else:
            logger.info("Evolution completed but no suitable programs were found.")
            return "Evolution completed but no suitable programs were found."
    except Exception as e:
        logger.error(f"Error in evolution process: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"
    finally:
        task_running = False

def format_program_for_display(program: Program) -> str:
    """Format a Program object for display in the UI."""
    result = f"Program ID: {program.id}\n"
    result += f"Generation: {program.generation}\n"
    
    if program.parent_id:
        result += f"Parent ID: {program.parent_id}\n"
    
    if program.fitness_scores:
        result += "Fitness Scores:\n"
        for key, value in program.fitness_scores.items():
            if key == "correctness" and isinstance(value, float):
                result += f"  {key}: {value * 100:.2f}%\n"
            else:
                result += f"  {key}: {value}\n"
    
    result += "\nCode:\n```python\n"
    result += program.code
    result += "\n```\n"
    
    return result

def load_example(example_key):
    """Load a predefined example task."""
    examples = {
        "shortest_path": {
            "id": "generic_shortest_path_problem",
            "description": "Given a weighted, directed graph and a starting node, find the shortest distance from the starting node to all other nodes in the graph. The graph is represented as a dictionary where keys are node identifiers (e.g., strings or integers), and values are dictionaries representing outgoing edges. In these inner dictionaries, keys are neighbor node identifiers and values are the weights (costs) of the edges to those neighbors. If a node is unreachable from the start node, its distance should be considered infinity. The function should return a dictionary where keys are node identifiers and values are the calculated shortest distances from the start node. The start node's distance to itself is 0.",
            "function": "solve_shortest_paths",
            "examples": """[
    {
        "input": [{"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}, "A"],
        "output": {"A": 0, "B": 1, "C": 3, "D": 4}
    },
    {
        "input": [{"A": {"B": 1}, "B": {"A": 2, "C": 5}, "C": {"D": 1}, "D": {}}, "A"],
        "output": {"A": 0, "B": 1, "C": 6, "D": 7}
    },
    {
        "input": [{"A": {"B": 1}, "B": {}, "C": {"D": 1}, "D": {}}, "A"],
        "output": {"A": 0, "B": 1, "C": "float('inf')", "D": "float('inf')"}
    }
]""",
            "imports": "heapq"
        },
        "sum_list": {
            "id": "sum_list_task",
            "description": "Write a Python function called `sum_numbers` that takes a list of integers `numbers` and returns their sum. The function should handle empty lists correctly by returning 0.",
            "function": "sum_numbers",
            "examples": """[
    {"input": [1, 2, 3], "output": 6},
    {"input": [], "output": 0},
    {"input": [-1, 0, 1], "output": 0},
    {"input": [10, 20, 30, 40, 50], "output": 150}
]""",
            "imports": ""
        },
        "fibonacci": {
            "id": "fibonacci_task",
            "description": "Implement a function `fibonacci` that calculates the nth Fibonacci number. The Fibonacci sequence starts with 0 and 1, and each subsequent number is the sum of the two preceding ones. For example, F(0) = 0, F(1) = 1, F(2) = 1, F(3) = 2, and so on.",
            "function": "fibonacci",
            "examples": """[
    {"input": 0, "output": 0},
    {"input": 1, "output": 1},
    {"input": 2, "output": 1},
    {"input": 3, "output": 2},
    {"input": 10, "output": 55},
    {"input": 20, "output": 6765}
]""",
            "imports": "functools"
        }
    }
    
    if example_key not in examples:
        return None, None, None, None, None
    
    example = examples[example_key]
    return (
        example["id"],
        example["description"],
        example["function"],
        example["examples"],
        example["imports"]
    )

# Gradio UI Definition
def build_ui():
    with gr.Blocks(title="OpenAlpha_Evolve Web Interface") as app:
        gr.Markdown("# OpenAlpha_Evolve: Autonomous Algorithmic Evolution")
        gr.Markdown("""
        This interface allows you to define algorithmic tasks and use Large Language Models (LLMs)
        to evolve solutions through an evolutionary process guided by the principles of natural selection.
        """)
        
        with gr.Tab("Task Definition"):
            with gr.Row():
                with gr.Column():
                    example_dropdown = gr.Dropdown(
                        choices=["shortest_path", "sum_list", "fibonacci"],
                        label="Load Example Task",
                        info="Select a predefined task to load its details.",
                        value=None
                    )
                    
                    task_id = gr.Textbox(
                        label="Task ID",
                        placeholder="Enter a unique identifier for this task (e.g., 'sorting_task')",
                        info="A unique identifier used to reference this task.",
                    )
                    
                    function_name = gr.Textbox(
                        label="Function Name to Evolve",
                        placeholder="Name of the function to generate (e.g., 'sort_array')",
                        info="The name of the Python function that the system will try to evolve.",
                    )
                    
                    allowed_imports = gr.Textbox(
                        label="Allowed Imports (Optional)",
                        placeholder="comma-separated list of allowed imports (e.g., 'heapq,collections,math')",
                        info="Python modules that the generated code is allowed to import.",
                    )
                    
                with gr.Column():
                    description = gr.Textbox(
                        label="Task Description",
                        placeholder="Provide a clear, detailed description of the problem...",
                        info="A natural language description of the problem. Be specific about what the function should do.",
                        lines=10
                    )
                    
                    examples = gr.Code(
                        label="Input/Output Examples (JSON format)",
                        language="json",
                        placeholder="""[
    {"input": [1, 2, 3], "output": 6},
    {"input": [], "output": 0}
]""",
                        info="A list of input/output examples. Each example should have 'input' and 'output' keys.",
                        lines=10
                    )
                    
            create_btn = gr.Button("Create Task Definition", variant="primary")
            task_status = gr.Markdown("Task not defined yet.")
            
        with gr.Tab("Evolution Process"):
            with gr.Row():
                with gr.Column():
                    population_size = gr.Slider(
                        minimum=3, maximum=20, value=settings.POPULATION_SIZE,
                        step=1, label="Population Size",
                        info="Number of programs in each generation."
                    )
                    
                    generations = gr.Slider(
                        minimum=1, maximum=10, value=settings.GENERATIONS,
                        step=1, label="Generations",
                        info="Number of generations to run the evolution."
                    )
                    
                    api_key = gr.Textbox(
                        label="Gemini API Key (Optional)",
                        placeholder="Enter your Gemini API key if not using .env file",
                        info="If provided, this will override the API key in your .env file.",
                        type="password"
                    )
                    
                with gr.Column():
                    update_settings_btn = gr.Button("Update Settings")
                    settings_status = gr.Markdown("Using current settings from config/settings.py")
                    
                    start_btn = gr.Button("Start Evolution", variant="primary")
                    stop_btn = gr.Button("Stop Evolution", variant="stop")
                    evolution_status = gr.Markdown("Evolution not started yet.")
            
        with gr.Tab("Results"):
            with gr.Row():
                with gr.Column():
                    result_selector = gr.Dropdown(
                        label="Select Solution",
                        choices=[],
                        info="Select a solution to view its details."
                    )
                    
                    refresh_btn = gr.Button("Refresh Results")
                    
                with gr.Column():
                    solution_details = gr.Markdown("No solutions available yet.")
        
        # Event handlers
        def update_settings(pop_size, gen_count, gemini_key):
            # Update settings
            if pop_size != settings.POPULATION_SIZE or gen_count != settings.GENERATIONS:
                settings.POPULATION_SIZE = pop_size
                settings.GENERATIONS = gen_count
            
            # Update API key if provided
            if gemini_key:
                settings.GEMINI_API_KEY = gemini_key
                return "Settings updated, including custom API key."
            
            return "Settings updated."
        
        async def create_task_and_update(id_val, desc_val, func_val, examples_val, imports_val):
            global current_task
            task, message = await create_task_definition(id_val, desc_val, func_val, examples_val, imports_val)
            if task:
                current_task = task
            return message
        
        async def start_evolution_process():
            global current_task, evolution_results
            
            if not current_task:
                return "Please create a task definition first."
            
            # Update temporary evolutionary parameters from settings
            # This ensures the values from UI sliders are used
            if not task_running:
                result = await run_evolution(current_task)
                return result
            
            return "Task is already running. Please wait or stop the current process."
        
        def get_result_choices():
            if not evolution_results:
                return [], "No solutions available yet."
            
            choices = [f"Solution {i+1} (ID: {prog.id})" for i, prog in enumerate(evolution_results)]
            return choices, "Select a solution from the dropdown."
        
        def display_solution(solution_idx):
            if not evolution_results or solution_idx is None:
                return "No solution selected."
            
            try:
                idx = int(solution_idx.split()[1]) - 1
                if 0 <= idx < len(evolution_results):
                    return format_program_for_display(evolution_results[idx])
                return "Invalid solution index."
            except:
                return "Error parsing solution index."
        
        # Connect event handlers
        example_dropdown.change(
            fn=load_example,
            inputs=example_dropdown,
            outputs=[task_id, description, function_name, examples, allowed_imports]
        )
        
        create_btn.click(
            fn=create_task_and_update,
            inputs=[task_id, description, function_name, examples, allowed_imports],
            outputs=task_status
        )
        
        update_settings_btn.click(
            fn=update_settings,
            inputs=[population_size, generations, api_key],
            outputs=settings_status
        )
        
        start_btn.click(
            fn=start_evolution_process,
            inputs=[],
            outputs=evolution_status
        )
        
        refresh_btn.click(
            fn=get_result_choices,
            inputs=[],
            outputs=[result_selector, solution_details]
        )
        
        result_selector.change(
            fn=display_solution,
            inputs=result_selector,
            outputs=solution_details
        )
        
    return app

if __name__ == "__main__":
    # Check if API key is set
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY.startswith("YOUR_API_KEY"):
        print("WARNING: Gemini API key not properly configured. Set it in .env file or provide it in the UI.")
    
    # Update requirements.txt to include gradio
    if not any(line.strip().startswith("gradio") for line in open("requirements.txt").readlines()):
        with open("requirements.txt", "a") as f:
            f.write("\n# Web Interface\ngradio\n")
    
    # Launch Gradio app
    app = build_ui()
    app.launch(share=True) 