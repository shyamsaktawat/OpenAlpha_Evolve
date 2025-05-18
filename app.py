"""
Gradio web interface for OpenAlpha_Evolve.
"""
import gradio as gr
import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()

from core.interfaces import TaskDefinition, Program
from task_manager.agent import TaskManagerAgent
from config import settings

# Setup a string handler to capture log messages
class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_capture = []
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_capture.append(msg)
        except Exception:
            self.handleError(record)
    
    def get_logs(self):
        return "\n".join(self.log_capture)
    
    def clear(self):
        self.log_capture = []

# Create a string handler
string_handler = StringIOHandler()
string_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add handler to root logger
root_logger = logging.getLogger()
root_logger.addHandler(string_handler)

# Also send logs to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set module loggers to DEBUG to get more information
for module in ['task_manager.agent', 'code_generator.agent', 'evaluator_agent.agent', 'database_agent.agent', 
              'selection_controller.agent', 'prompt_designer.agent']:
    logging.getLogger(module).setLevel(logging.DEBUG)

# Check if API key is set
if settings.GEMINI_API_KEY.startswith("YOUR_API_KEY") or not settings.GEMINI_API_KEY:
    API_KEY_WARNING = "⚠️ API key not properly set! Please set your Gemini API key in the .env file."
else:
    API_KEY_WARNING = ""

# Global variables for storing evolution state
current_results = []

async def run_evolution(
    task_id, 
    description, 
    function_name, 
    examples_json, 
    allowed_imports_text,
    population_size, 
    generations
):
    """Run the evolutionary process with the given parameters."""
    progress = gr.Progress()
    # Clear previous logs
    string_handler.clear()
    
    try:
        # Parse the input/output examples
        try:
            examples = json.loads(examples_json)
            if not isinstance(examples, list):
                return "Error: Examples must be a JSON list of objects with 'input' and 'output' keys."
            
            # Validate each example
            for i, example in enumerate(examples):
                if not isinstance(example, dict) or "input" not in example or "output" not in example:
                    return f"Error in example {i+1}: Each example must be an object with 'input' and 'output' keys."
        except json.JSONDecodeError:
            return "Error: Examples must be valid JSON. Please check the format."
        
        # Parse allowed imports
        allowed_imports = [imp.strip() for imp in allowed_imports_text.split(",") if imp.strip()]
        
        # Update settings from UI
        settings.POPULATION_SIZE = int(population_size)
        settings.GENERATIONS = int(generations)
        
        # Create a task definition
        task = TaskDefinition(
            id=task_id,
            description=description,
            function_name_to_evolve=function_name,
            input_output_examples=examples,
            allowed_imports=allowed_imports
        )
        
        # Set up a progress callback
        async def progress_callback(generation, max_generations, stage, message=""):
            # Calculate progress based on generation and stage
            # Stages: 0=init, 1=evaluation, 2=selection, 3=reproduction
            stage_weight = 0.25  # Each stage is worth 25% of a generation
            gen_progress = generation + (stage * stage_weight)
            total_progress = gen_progress / max_generations
            
            # Update the progress bar
            progress(min(total_progress, 0.99), f"Generation {generation}/{max_generations}: {message}")
            
            # Also log the progress
            logger.info(f"Progress: Generation {generation}/{max_generations} - {message}")
            
            # Allow the UI to update
            await asyncio.sleep(0.1)
        
        # Initialize the TaskManagerAgent with the task definition
        task_manager = TaskManagerAgent(task_definition=task)
        
        # Add a custom attribute to track progress (doesn't affect the class behavior)
        task_manager.progress_callback = progress_callback
        
        # Execute the evolutionary process with progress updates
        progress(0, "Starting evolutionary process...")
        
        # First listener setup to catch log messages about generations
        class GenerationProgressListener(logging.Handler):
            def __init__(self):
                super().__init__()
                self.current_gen = 0
                self.max_gen = settings.GENERATIONS
                
            def emit(self, record):
                try:
                    msg = record.getMessage()
                    # Check for generation progress messages
                    if "--- Generation " in msg:
                        gen_parts = msg.split("Generation ")[1].split("/")[0]
                        try:
                            self.current_gen = int(gen_parts)
                            # Update progress bar
                            asyncio.create_task(
                                progress_callback(
                                    self.current_gen, 
                                    self.max_gen, 
                                    0, 
                                    "Starting generation"
                                )
                            )
                        except ValueError:
                            pass
                    elif "Evaluating population" in msg:
                        # Update progress for evaluation stage
                        asyncio.create_task(
                            progress_callback(
                                self.current_gen, 
                                self.max_gen, 
                                1, 
                                "Evaluating population"
                            )
                        )
                    elif "Selected " in msg and " parents" in msg:
                        # Update progress for selection stage
                        asyncio.create_task(
                            progress_callback(
                                self.current_gen, 
                                self.max_gen, 
                                2, 
                                "Selected parents"
                            )
                        )
                    elif "Generated " in msg and " offspring" in msg:
                        # Update progress for reproduction stage
                        asyncio.create_task(
                            progress_callback(
                                self.current_gen, 
                                self.max_gen, 
                                3, 
                                "Generated offspring"
                            )
                        )
                except Exception:
                    pass
        
        # Add our progress listener
        progress_listener = GenerationProgressListener()
        progress_listener.setLevel(logging.INFO)
        root_logger.addHandler(progress_listener)
        
        try:
            # Execute the evolutionary process
            best_programs = await task_manager.execute()
            progress(1.0, "Evolution completed!")
            
            # Store results for display
            global current_results
            current_results = best_programs if best_programs else []
            
            # Format results
            if best_programs:
                result_text = f"✅ Evolution completed successfully! Found {len(best_programs)} solution(s).\n\n"
                for i, program in enumerate(best_programs):
                    result_text += f"### Solution {i+1}\n"
                    result_text += f"- ID: {program.id}\n"
                    result_text += f"- Fitness: {program.fitness_scores}\n"
                    result_text += f"- Generation: {program.generation}\n\n"
                    result_text += "```python\n" + program.code + "\n```\n\n"
                return result_text
            else:
                return "❌ Evolution completed, but no suitable solutions were found."
        finally:
            # Remove our progress listener when done
            root_logger.removeHandler(progress_listener)
    
    except Exception as e:
        import traceback
        return f"Error during evolution: {str(e)}\n\n{traceback.format_exc()}"

def get_code(solution_index):
    """Get the code for a specific solution."""
    try:
        if current_results and 0 <= solution_index < len(current_results):
            program = current_results[solution_index]
            return program.code
        return "No solution available at this index."
    except Exception as e:
        return f"Error retrieving solution: {str(e)}"

# Example templates: Fibonacci task
FIB_EXAMPLES = '''[
    {"input": [0], "output": 0},
    {"input": [1], "output": 1},
    {"input": [5], "output": 5},
    {"input": [10], "output": 55}
]'''

def set_fib_example():
    """Populate the form with the Fibonacci task example."""
    return (
        "fibonacci_task",
        "Write a Python function that computes the nth Fibonacci number (0-indexed), where fib(0)=0 and fib(1)=1.",
        "fibonacci",
        FIB_EXAMPLES,
        ""
    )

# Create the Gradio interface
with gr.Blocks(title="OpenAlpha_Evolve") as demo:
    gr.Markdown("# 🧬 OpenAlpha_Evolve: AI-Driven Algorithm Evolution")
    
    if API_KEY_WARNING:
        gr.Markdown(f"## ⚠️ {API_KEY_WARNING}")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Task Definition")
            
            task_id = gr.Textbox(
                label="Task ID", 
                placeholder="e.g., fibonacci_task",
                value="fibonacci_task"
            )
            
            description = gr.Textbox(
                label="Task Description", 
                placeholder="Describe the problem clearly...",
                value="Write a Python function that computes the nth Fibonacci number (0-indexed), where fib(0)=0 and fib(1)=1.",
                lines=5
            )
            
            function_name = gr.Textbox(
                label="Function Name to Evolve", 
                placeholder="e.g., fibonacci",
                value="fibonacci"
            )
            
            examples_json = gr.Code(
                label="Input/Output Examples (JSON)",
                language="json",
                value=FIB_EXAMPLES,
                lines=10
            )
            
            allowed_imports = gr.Textbox(
                label="Allowed Imports (comma-separated)",
                placeholder="e.g., math",
                value=""
            )
            
            with gr.Row():
                population_size = gr.Slider(
                    label="Population Size",
                    minimum=2, 
                    maximum=10, 
                    value=3, 
                    step=1
                )
                
                generations = gr.Slider(
                    label="Generations",
                    minimum=1, 
                    maximum=5, 
                    value=2, 
                    step=1
                )
            
            with gr.Row():
                fib_btn = gr.Button("🔢 Fibonacci Example")
                run_btn = gr.Button("🚀 Run Evolution", variant="primary")
        
        with gr.Column(scale=1):
            with gr.Tab("Results"):
                results_text = gr.Markdown("Evolution results will appear here...")
            
            # No Live Logs tab: progress is shown in terminal only
    
    # Event handlers
    # Example setter for Fibonacci
    fib_btn.click(
        set_fib_example,
        outputs=[task_id, description, function_name, examples_json, allowed_imports]
    )
    
    run_evolution_event = run_btn.click(
        run_evolution,
        inputs=[
            task_id, 
            description, 
            function_name, 
            examples_json,
            allowed_imports,
            population_size, 
            generations
        ],
        outputs=results_text
    )

# Launch the app
if __name__ == "__main__":
    # Launch with share=True to create a public link
    demo.launch(share=True) 