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

                                               
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

                                           
load_dotenv()

from core.interfaces import TaskDefinition, Program
from task_manager.agent import TaskManagerAgent
from config import settings

                                                
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

                         
string_handler = StringIOHandler()
string_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

                            
root_logger = logging.getLogger()
root_logger.addHandler(string_handler)

                           
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

                                   
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

                                                     
for module in ['task_manager.agent', 'code_generator.agent', 'evaluator_agent.agent', 'database_agent.agent', 
              'selection_controller.agent', 'prompt_designer.agent']:
    logging.getLogger(module).setLevel(logging.DEBUG)

                         

                                              
current_results = []

async def run_evolution(
    task_id, 
    description, 
    function_name, 
    examples_json, 
    allowed_imports_text,
    population_size, 
    generations,
    num_islands,
    migration_frequency,
    migration_rate
):
    """Run the evolutionary process with the given parameters."""
    progress = gr.Progress()
                         
    string_handler.clear()
    
    try:
                                         
        try:
            examples = json.loads(examples_json)
            if not isinstance(examples, list):
                return "Error: Examples must be a JSON list of objects with 'input' and 'output' keys."
            
                                   
            for i, example in enumerate(examples):
                if not isinstance(example, dict) or "input" not in example or "output" not in example:
                    return f"Error in example {i+1}: Each example must be an object with 'input' and 'output' keys."
        except json.JSONDecodeError:
            return "Error: Examples must be valid JSON. Please check the format."
        
                               
        allowed_imports = [imp.strip() for imp in allowed_imports_text.split(",") if imp.strip()]
        
                                 
        settings.POPULATION_SIZE = int(population_size)
        settings.GENERATIONS = int(generations)
        settings.NUM_ISLANDS = int(num_islands)
        settings.MIGRATION_FREQUENCY = int(migration_frequency)
        settings.MIGRATION_RATE = float(migration_rate)
        
                                  
        task = TaskDefinition(
            id=task_id,
            description=description,
            function_name_to_evolve=function_name,
            input_output_examples=examples,
            allowed_imports=allowed_imports
        )
        
                                    
        async def progress_callback(generation, max_generations, stage, message=""):
                                                              
                                                                       
            stage_weight = 0.25                                           
            gen_progress = generation + (stage * stage_weight)
            total_progress = gen_progress / max_generations
            
                                     
            progress(min(total_progress, 0.99), f"Generation {generation}/{max_generations}: {message}")
            
                                   
            logger.info(f"Progress: Generation {generation}/{max_generations} - {message}")
            
                                    
            await asyncio.sleep(0.1)
        
                                                                  
        task_manager = TaskManagerAgent(task_definition=task)
        
                                                                                      
        task_manager.progress_callback = progress_callback
        
                                                                
        progress(0, "Starting evolutionary process...")
        
                                                                      
        class GenerationProgressListener(logging.Handler):
            def __init__(self):
                super().__init__()
                self.current_gen = 0
                self.max_gen = settings.GENERATIONS
                
            def emit(self, record):
                try:
                    msg = record.getMessage()
                                                            
                    if "--- Generation " in msg:
                        gen_parts = msg.split("Generation ")[1].split("/")[0]
                        try:
                            self.current_gen = int(gen_parts)
                                                 
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
                                                              
                        asyncio.create_task(
                            progress_callback(
                                self.current_gen, 
                                self.max_gen, 
                                1, 
                                "Evaluating population"
                            )
                        )
                    elif "Selected " in msg and " parents" in msg:
                                                             
                        asyncio.create_task(
                            progress_callback(
                                self.current_gen, 
                                self.max_gen, 
                                2, 
                                "Selected parents"
                            )
                        )
                    elif "Generated " in msg and " offspring" in msg:
                                                                
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
        
                                   
        progress_listener = GenerationProgressListener()
        progress_listener.setLevel(logging.INFO)
        root_logger.addHandler(progress_listener)
        
        try:
                                              
            best_programs = await task_manager.execute()
            progress(1.0, "Evolution completed!")
            
                                       
            global current_results
            current_results = best_programs if best_programs else []
            
                            
            if best_programs:
                result_text = f"✅ Evolution completed successfully! Found {len(best_programs)} solution(s).\n\n"
                for i, program in enumerate(best_programs):
                    result_text += f"### Solution {i+1}\n"
                    result_text += f"- ID: {program.id}\n"
                    result_text += f"- Fitness: {program.fitness_scores}\n"
                    result_text += f"- Generation: {program.generation}\n"
                    result_text += f"- Island ID: {program.island_id}\n\n"
                    result_text += "```python\n" + program.code + "\n```\n\n"
                return result_text
            else:
                return "❌ Evolution completed, but no suitable solutions were found."
        finally:
                                                    
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

                                   
FIB_EXAMPLES = '''[
    {"input": [0], "output": 0},
    {"input": [1], "output": 1},
    {"input": [5], "output": 5},
    {"input": [10], "output": 55}
]'''

def set_fib_example():
    """Set the UI to a Fibonacci example task."""
    return (
        "fibonacci_task",
        "Write a Python function that computes the nth Fibonacci number (0-indexed), where fib(0)=0 and fib(1)=1.",
        "fibonacci",
        FIB_EXAMPLES,
        ""
    )

                             
with gr.Blocks(title="OpenAlpha_Evolve") as demo:
    gr.Markdown("# 🧬 OpenAlpha_Evolve: Autonomous Algorithm Evolution")
    gr.Markdown("""
    * **Custom Tasks:** Write your own problem definition, examples, and allowed imports in the fields below.
    * **Multi-Model Support:** Additional language model backends coming soon.
    * **Evolutionary Budget:** For novel, complex solutions consider using large budgets (e.g., 100+ generations and population sizes of hundreds or thousands).
    * **Island Model:** The population is divided into islands that evolve independently, with periodic migration between them.
    """)
    
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
                num_islands = gr.Slider(
                    label="Number of Islands",
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1
                )
                
                migration_frequency = gr.Slider(
                    label="Migration Frequency (generations)",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1
                )
                
                migration_rate = gr.Slider(
                    label="Migration Rate",
                    minimum=0.1,
                    maximum=0.5,
                    value=0.2,
                    step=0.1
                )
            
            with gr.Row():
                example_btn = gr.Button("📘 Fibonacci Example")
            
            run_btn = gr.Button("🚀 Run Evolution", variant="primary")
        
        with gr.Column(scale=1):
            with gr.Tab("Results"):
                results_text = gr.Markdown("Evolution results will appear here...")
            
                                                                  
    
                    
    example_btn.click(
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
            generations,
            num_islands,
            migration_frequency,
            migration_rate
        ],
        outputs=results_text
    )

                
if __name__ == "__main__":
                                                    
    demo.launch(share=True) 