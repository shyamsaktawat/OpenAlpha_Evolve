import os
import sys
import json
import asyncio
import logging
import ast
import time
import threading
import gradio as gr
from typing import Dict, Any, List, Optional, Tuple, Callable

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

# Create a custom handler that will capture logs for the UI
class UILogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.lock = threading.Lock()
        
    def emit(self, record):
        with self.lock:
            self.logs.append({
                'time': record.asctime if hasattr(record, 'asctime') else time.strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'name': record.name,
                'message': self.format(record)
            })
            # Keep only the last 1000 logs to avoid memory issues
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
                
    def get_logs(self, level=None):
        with self.lock:
            if level:
                return [log for log in self.logs if log['level'] == level.upper()]
            return self.logs.copy()
            
    def clear(self):
        with self.lock:
            self.logs = []

# Add custom handler to root logger
ui_log_handler = UILogHandler()
ui_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(ui_log_handler)

# Global variables to store current task and results
current_task = None
evolution_results = []
task_running = False
stop_requested = False

# Dictionary of supported LLM API providers
API_PROVIDERS = {
    "gemini": {
        "name": "Google Gemini",
        "env_var": "GEMINI_API_KEY",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro-vision"],
        "api_key_url": "https://aistudio.google.com/app/apikey",
        "config_vars": ["GEMINI_PRO_MODEL_NAME", "GEMINI_FLASH_MODEL_NAME", "GEMINI_EVALUATION_MODEL"]
    },
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
        "api_key_url": "https://platform.openai.com/api-keys",
        "config_vars": []
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "api_key_url": "https://console.anthropic.com/keys",
        "config_vars": []
    }
}

# Currently, only Gemini is supported in the code. 
# This interface prepares for future extensions to other providers.
ACTIVE_PROVIDERS = ["gemini"]

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
) -> Tuple[Optional[TaskDefinition], str]:
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
    
    logger.info(f"Created task definition: '{task_id}' with {len(examples)} examples")
    return task, "Task definition created successfully."

async def update_evolution_status(status_callback: Callable, interval: float = 1.0):
    """Continuously update the evolution status in the UI based on captured logs."""
    global task_running, stop_requested
    
    while task_running and not stop_requested:
        logs = ui_log_handler.get_logs()
        
        if logs:
            # Create a formatted log display with the most recent logs
            log_text = "\n".join([f"{log['time']} - {log['level']} - {log['message']}" 
                                for log in logs[-20:]])  # Show last 20 logs
            
            # Extract key metrics if available
            generation = "0"
            best_fitness = "N/A"
            population_size = str(settings.POPULATION_SIZE)
            
            for log in reversed(logs):  # Check most recent logs first
                message = log['message']
                if "Generation" in message and "Best program" in message:
                    gen_parts = message.split("Generation")
                    if len(gen_parts) > 1:
                        gen_num = gen_parts[1].strip().split(":")[0].strip()
                        if gen_num.isdigit():
                            generation = gen_num
                    
                    if "Fitness=" in message:
                        fitness_parts = message.split("Fitness=")
                        if len(fitness_parts) > 1:
                            best_fitness = fitness_parts[1].strip()
                    break
            
            status_html = f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
                <h3>Evolution Status</h3>
                <p><b>Status:</b> Running</p>
                <p><b>Generation:</b> {generation} of {settings.GENERATIONS}</p>
                <p><b>Population Size:</b> {population_size}</p>
                <p><b>Best Fitness:</b> {best_fitness}</p>
                <div style="max-height: 300px; overflow-y: auto; background: #eee; padding: 10px; font-family: monospace; white-space: pre-wrap;">
                {log_text}
                </div>
            </div>
            """
            
            status_callback(status_html)
        
        await asyncio.sleep(interval)
    
    # Final update after completion
    status_callback(f"""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
        <h3>Evolution Status</h3>
        <p><b>Status:</b> {"Stopped by user" if stop_requested else "Completed"}</p>
        <p><b>Solutions found:</b> {len(evolution_results)}</p>
        <p>Check the Results tab to see the evolved solutions.</p>
    </div>
    """)

async def run_evolution(task: TaskDefinition, status_callback=None):
    """Run the evolutionary process on the defined task."""
    global task_running, evolution_results, stop_requested
    
    if task_running:
        return "Task is already running. Please wait."
    
    task_running = True
    stop_requested = False
    evolution_results = []
    ui_log_handler.clear()
    
    # Start a background task to update the UI
    if status_callback:
        asyncio.create_task(update_evolution_status(status_callback))
    
    try:
        logger.info(f"Starting evolution process for task: {task.id}")
        logger.info(f"Settings: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")
        
        task_manager = TaskManagerAgent(task_definition=task)
        
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

def stop_evolution():
    """Signal that the evolution process should stop."""
    global stop_requested
    
    if task_running:
        stop_requested = True
        logger.info("Stopping evolution process (will complete current generation)...")
        return "Stopping evolution process. Please wait for current operations to complete."
    
    return "No evolution process is currently running."

def format_program_for_display(program: Program) -> str:
    """Format a Program object for display in the UI."""
    result = f"""
    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px;">
        <h3>Program Details</h3>
        <p><b>ID:</b> {program.id}</p>
        <p><b>Generation:</b> {program.generation}</p>
    """
    
    if program.parent_id:
        result += f"<p><b>Parent ID:</b> {program.parent_id}</p>"
    
    if program.fitness_scores:
        result += "<h4>Fitness Scores</h4><ul>"
        for key, value in program.fitness_scores.items():
            if key == "correctness" and isinstance(value, float):
                result += f"<li><b>{key}:</b> {value * 100:.2f}%</li>"
            else:
                result += f"<li><b>{key}:</b> {value}</li>"
        result += "</ul>"
    
    if program.errors:
        result += "<h4>Errors</h4><ul>"
        for error in program.errors[:5]:  # Show at most 5 errors
            result += f"<li>{error}</li>"
        if len(program.errors) > 5:
            result += f"<li>... and {len(program.errors) - 5} more errors</li>"
        result += "</ul>"
    
    result += """
        <h4>Code</h4>
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
            <pre><code>
"""
    result += program.code.replace("<", "&lt;").replace(">", "&gt;")
    result += """
            </code></pre>
        </div>
    </div>
    """
    
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
        },
        "sorting": {
            "id": "sorting_task",
            "description": "Implement a function `custom_sort` that sorts a list of integers in ascending order. You may implement any sorting algorithm of your choice (e.g., bubble sort, merge sort, quicksort). The function should return a new sorted list without modifying the original input list.",
            "function": "custom_sort",
            "examples": """[
    {"input": [5, 2, 9, 1, 5, 6], "output": [1, 2, 5, 5, 6, 9]},
    {"input": [], "output": []},
    {"input": [1], "output": [1]},
    {"input": [3, 1, 4, 1, 5, 9, 2, 6, 5], "output": [1, 1, 2, 3, 4, 5, 5, 6, 9]},
    {"input": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "output": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
]""",
            "imports": ""
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

def update_api_settings(provider, api_key, model_choices):
    """Update API settings based on UI selections."""
    if provider not in ACTIVE_PROVIDERS:
        return f"Provider {provider} is not currently supported for evolution. Only {', '.join(ACTIVE_PROVIDERS)} supported."
    
    if not api_key:
        return "Please provide an API key."
    
    # Update the appropriate settings variables based on provider
    if provider == "gemini":
        settings.GEMINI_API_KEY = api_key
        
        # Update model selections if provided
        if len(model_choices) >= 3:
            settings.GEMINI_PRO_MODEL_NAME = model_choices[0]
            settings.GEMINI_FLASH_MODEL_NAME = model_choices[1]
            settings.GEMINI_EVALUATION_MODEL = model_choices[2]
            
            return f"Updated Gemini API key and model selections: {model_choices}"
        else:
            return "Updated Gemini API key."
    
    # For future implementation of other providers
    return f"Updated {provider} API key."

# Gradio UI Definition
def build_ui():
    with gr.Blocks(title="OpenAlpha_Evolve Web Interface", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🧬 OpenAlpha_Evolve: Autonomous Algorithmic Evolution")
        gr.Markdown("""
        This interface allows you to define algorithmic tasks and use Large Language Models (LLMs)
        to evolve solutions through an evolutionary process guided by the principles of natural selection.
        
        OpenAlpha_Evolve iteratively generates, tests, and improves code to solve your defined problem.
        """)
        
        with gr.Tabs() as tabs:
            with gr.Tab("📋 Task Definition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        example_dropdown = gr.Dropdown(
                            choices=["shortest_path", "sum_list", "fibonacci", "sorting"],
                            label="📚 Load Example Task",
                            info="Select a predefined task to load its details.",
                            value=None
                        )
                        
                        task_id = gr.Textbox(
                            label="🏷️ Task ID",
                            placeholder="Enter a unique identifier for this task (e.g., 'sorting_task')",
                            info="A unique identifier used to reference this task.",
                        )
                        
                        function_name = gr.Textbox(
                            label="⚙️ Function Name to Evolve",
                            placeholder="Name of the function to generate (e.g., 'sort_array')",
                            info="The name of the Python function that the system will try to evolve.",
                        )
                        
                        allowed_imports = gr.Textbox(
                            label="📦 Allowed Imports (Optional)",
                            placeholder="comma-separated list of allowed imports (e.g., 'heapq,collections,math')",
                            info="Python modules that the generated code is allowed to import.",
                        )
                        
                    with gr.Column(scale=2):
                        description = gr.Textbox(
                            label="📝 Task Description",
                            placeholder="Provide a clear, detailed description of the problem...",
                            info="A natural language description of the problem. Be specific about what the function should do.",
                            lines=10
                        )
                        
                        examples = gr.Code(
                            label="🧪 Input/Output Examples (JSON format)",
                            language="json",
                            placeholder="""[
    {"input": [1, 2, 3], "output": 6},
    {"input": [], "output": 0}
]""",
                            info="A list of input/output examples. Each example should have 'input' and 'output' keys.",
                            lines=10
                        )
                        
                with gr.Row():
                    create_btn = gr.Button("✅ Create Task Definition", variant="primary", scale=1)
                    clear_task_btn = gr.Button("🗑️ Clear Fields", scale=1)
                    
                task_status = gr.Markdown("Task not defined yet.")
                
            with gr.Tab("⚙️ API Configuration"):
                with gr.Row():
                    with gr.Column():
                        api_provider = gr.Dropdown(
                            choices=[provider for provider in API_PROVIDERS.keys() if provider in ACTIVE_PROVIDERS],
                            label="🔌 API Provider",
                            info="Select the LLM provider to use for code generation and evaluation.",
                            value="gemini"
                        )
                        
                        api_key = gr.Textbox(
                            label="🔑 API Key",
                            placeholder="Enter your API key",
                            info="Your API key for the selected provider.",
                            type="password"
                        )
                        
                        api_link = gr.Markdown("Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
                        
                    with gr.Column():
                        with gr.Box():
                            gr.Markdown("### 🤖 Model Selection")
                            
                            gen_model = gr.Dropdown(
                                label="Generation Model",
                                choices=API_PROVIDERS["gemini"]["models"],
                                info="Model for primary code generation.",
                                value=settings.GEMINI_PRO_MODEL_NAME
                            )
                            
                            fast_model = gr.Dropdown(
                                label="Fast Model",
                                choices=API_PROVIDERS["gemini"]["models"],
                                info="Model for quicker, less complex tasks.",
                                value=settings.GEMINI_FLASH_MODEL_NAME
                            )
                            
                            eval_model = gr.Dropdown(
                                label="Evaluation Model",
                                choices=API_PROVIDERS["gemini"]["models"],
                                info="Model for evaluating program fitness.",
                                value=settings.GEMINI_EVALUATION_MODEL
                            )
                            
                            update_api_btn = gr.Button("💾 Save API Configuration", variant="primary")
                
                api_status = gr.Markdown("Using default API configuration from config/settings.py")
                current_config = gr.Markdown(f"""
                **Current Configuration:**
                - API Provider: Gemini
                - Generation Model: {settings.GEMINI_PRO_MODEL_NAME}
                - Fast Model: {settings.GEMINI_FLASH_MODEL_NAME} 
                - Evaluation Model: {settings.GEMINI_EVALUATION_MODEL}
                """)
                
            with gr.Tab("🧬 Evolution Process"):
                with gr.Row():
                    with gr.Column():
                        population_size = gr.Slider(
                            minimum=3, maximum=20, value=settings.POPULATION_SIZE,
                            step=1, label="👥 Population Size",
                            info="Number of programs in each generation."
                        )
                        
                        generations = gr.Slider(
                            minimum=1, maximum=10, value=settings.GENERATIONS,
                            step=1, label="🔄 Generations",
                            info="Number of generations to run the evolution."
                        )
                        
                        mutation_rate = gr.Slider(
                            minimum=0.1, maximum=1.0, value=settings.MUTATION_RATE,
                            step=0.1, label="🧬 Mutation Rate",
                            info="Probability of mutating an individual."
                        )
                        
                        update_params_btn = gr.Button("💾 Update Evolution Parameters")
                        params_status = gr.Markdown("Using current settings from config/settings.py")
                        
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Evolution", variant="primary", scale=2)
                    stop_btn = gr.Button("⏹️ Stop Evolution", variant="stop", scale=1)
                
                evolution_status = gr.HTML("Evolution not started yet.", elem_id="evolution_status")
            
            with gr.Tab("📊 Results"):
                with gr.Row():
                    with gr.Column(scale=1):
                        result_selector = gr.Dropdown(
                            label="📋 Select Solution",
                            choices=[],
                            info="Select a solution to view its details."
                        )
                        
                        refresh_btn = gr.Button("🔄 Refresh Results")
                        export_btn = gr.Button("📥 Export Selected Solution")
                        
                    with gr.Column(scale=3):
                        solution_details = gr.HTML("No solutions available yet.")
                
                download_code = gr.File(label="Download Code", visible=False)
        
        # Event handlers
        def update_api_provider(provider):
            """Update displayed information based on selected provider"""
            if provider in API_PROVIDERS:
                api_url = API_PROVIDERS[provider]["api_key_url"]
                provider_name = API_PROVIDERS[provider]["name"]
                
                return f"Get your {provider_name} API key from [{provider_name}]({api_url})"
            return ""
        
        def update_evolution_params(pop_size, gen_count, mut_rate):
            """Update evolution parameters settings"""
            settings.POPULATION_SIZE = pop_size
            settings.GENERATIONS = gen_count
            settings.MUTATION_RATE = mut_rate
            return f"Updated evolution parameters: Population={pop_size}, Generations={gen_count}, Mutation Rate={mut_rate}"
        
        def clear_task_fields():
            """Clear all task definition fields"""
            return None, "", "", "", ""
            
        def update_api_config(provider, key, gen_model, fast_model, eval_model):
            """Update API settings and return current configuration"""
            result = update_api_settings(provider, key, [gen_model, fast_model, eval_model])
            
            # Generate current configuration display
            if provider == "gemini":
                config_text = f"""
                **Current Configuration:**
                - API Provider: {API_PROVIDERS[provider]['name']}
                - Generation Model: {settings.GEMINI_PRO_MODEL_NAME}
                - Fast Model: {settings.GEMINI_FLASH_MODEL_NAME} 
                - Evaluation Model: {settings.GEMINI_EVALUATION_MODEL}
                """
                return result, config_text
            
            return result, "Configuration updated."
        
        async def create_task_and_update(id_val, desc_val, func_val, examples_val, imports_val):
            """Create a task definition and update the global current_task"""
            global current_task
            task, message = await create_task_definition(id_val, desc_val, func_val, examples_val, imports_val)
            if task:
                current_task = task
            return message
        
        async def start_evolution_process():
            """Start the evolutionary process if a task has been defined"""
            global current_task, evolution_results
            
            if not current_task:
                return "Please create a task definition first."
            
            # Update temporary evolutionary parameters from settings
            # This ensures the values from UI sliders are used
            if not task_running:
                return await run_evolution(current_task, lambda x: evolution_status.update(x))
            
            return "Task is already running. Please wait or stop the current process."
        
        def get_result_choices():
            """Get list of available solutions for the dropdown"""
            if not evolution_results:
                return [], "No solutions available yet."
            
            choices = [f"Solution {i+1} (ID: {prog.id})" for i, prog in enumerate(evolution_results)]
            return choices, "Select a solution from the dropdown."
        
        def display_solution(solution_idx):
            """Display details of the selected solution"""
            if not evolution_results or solution_idx is None:
                return "No solution selected."
            
            try:
                idx = int(solution_idx.split()[1]) - 1
                if 0 <= idx < len(evolution_results):
                    return format_program_for_display(evolution_results[idx])
                return "Invalid solution index."
            except:
                return "Error parsing solution index."
        
        def export_solution(solution_idx):
            """Export the selected solution as a Python file"""
            if not evolution_results or solution_idx is None:
                return None
            
            try:
                idx = int(solution_idx.split()[1]) - 1
                if 0 <= idx < len(evolution_results):
                    program = evolution_results[idx]
                    
                    # Create a temporary file
                    file_path = f"{program.id}.py"
                    with open(file_path, "w") as f:
                        f.write(f"# {program.id}\n")
                        f.write(f"# Generated by OpenAlpha_Evolve\n")
                        f.write(f"# Generation: {program.generation}\n")
                        f.write(f"# Fitness: {program.fitness_scores}\n\n")
                        f.write(program.code)
                    
                    return file_path
                    
            except Exception as e:
                logger.error(f"Error exporting solution: {e}")
                
            return None
            
        # Connect event handlers
        api_provider.change(
            fn=update_api_provider,
            inputs=api_provider,
            outputs=api_link
        )
        
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
        
        clear_task_btn.click(
            fn=clear_task_fields,
            inputs=[],
            outputs=[example_dropdown, task_id, description, function_name, examples, allowed_imports]
        )
        
        update_api_btn.click(
            fn=update_api_config,
            inputs=[api_provider, api_key, gen_model, fast_model, eval_model],
            outputs=[api_status, current_config]
        )
        
        update_params_btn.click(
            fn=update_evolution_params,
            inputs=[population_size, generations, mutation_rate],
            outputs=params_status
        )
        
        start_btn.click(
            fn=start_evolution_process,
            inputs=[],
            outputs=evolution_status
        )
        
        stop_btn.click(
            fn=stop_evolution,
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
        
        export_btn.click(
            fn=export_solution,
            inputs=result_selector,
            outputs=download_code
        )
        
    return app

if __name__ == "__main__":
    # Check if API key is set
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY.startswith("YOUR_API_KEY"):
        print("WARNING: Gemini API key not properly configured. Set it in .env file or provide it in the UI.")
    
    # Update requirements.txt to include gradio
    if not any(line.strip().startswith("gradio") for line in open("requirements.txt").readlines()):
        with open("requirements.txt", "a") as f:
            f.write("\n# Web Interface\ngradio>=3.40.0\n")
    
    # Launch Gradio app
    app = build_ui()
    app.launch(share=True, inbrowser=True) 