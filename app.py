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
from gradio.themes import Ocean
from translations import translations
import locale
from newTheme import DarkEvolveV2

# 👇 ADICIONADO: Função para determinar o idioma inicial
def get_initial_lang():
    """Detecta o idioma do sistema ou usa 'en' como padrão."""
    try:
        system_lang = locale.getlocale()[0] or "en_US"
        if system_lang:
            lang_code = system_lang.split('_')[0]
            if lang_code in translations:
                return lang_code
    except Exception:
        pass
    return 'en'

initial_lang = get_initial_lang()
                                               
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
theme = DarkEvolveV2()

                             
with gr.Blocks(title="OpenAlpha_Evolve", theme=theme, css="""
    .mybox{
        border: 1px solid #212534;
        padding: 20px;
        border-radius: 10px;
    }
    .form{
        background: none;
        border: none;
                 box-shadow: 0 0 0 #000;
    }
    .block{
        background: none;
               box-shadow:0;
    }
    .botaoExemplo{
        background-color: #121a2e;
          border-radius: 6px;
               }
    .enviar{
        background-color: #0f4fcf;
        color: #ffffff;
          border-radius: 6px;
               }
    .mySLider{
               border: 1px solid #212534;}
               .gradio-container {background-color: #0f121a}
""") as demo:
    t = translations[initial_lang]
        
    gr.Markdown(f"# {t['title']}")
    gr.Markdown(t['subtitle'])  

            # with gr.Column(scale=1, min_width=150):
    
    with gr.Row():
        with gr.Column(scale=5, elem_classes="mybox"):
            gr.Markdown("## Task Definition")
            
            task_id = gr.Textbox(
                label=t["task_id"], 
                placeholder=t["task_id_placeholder"],
                value="fibonacci_task"
            )
            
            description = gr.Textbox(
                label=t['description'], 
                placeholder=t['description_placeholder'],
                value="Write a Python function that computes the nth Fibonacci number (0-indexed), where fib(0)=0 and fib(1)=1.",
                lines=5
            )
            
            function_name = gr.Textbox(
                label=t['function_name'], 
                placeholder=t['function_name_placeholder'],
                value="fibonacci"
            )
            
            allowed_imports = gr.Textbox(
                label=t['allowed_imports'],
                placeholder=t['allowed_imports_placeholder'],
                value=""
            )
            examples_json = gr.Code(
                label=t['examples_json'],
                language="json",
                value=FIB_EXAMPLES,
                lines=10
            )
            
        with gr.Column(scale=2, elem_classes="mybox"):
            gr.Markdown("# Configurations")

                    
            population_size = gr.Slider(
                    label=t['population_size'],
                    minimum=2, 
                    maximum=10, 
                    value=3, 
                    step=1,
                    elem_classes='mySLider'
                    )
                    
            generations = gr.Slider(
                        label=t['generations'],
                        minimum=1, 
                        maximum=5, 
                        value=2, 
                        step=1
                    )
                
               
            num_islands = gr.Slider(
                        label=t['num_islands'],
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1
                    )
                    
            migration_frequency = gr.Slider(
                        label=t['migration_frequency'],
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1
                    )
                    
            migration_rate = gr.Slider(
                        label=t['migration_rate'],
                        minimum=0.1,
                        maximum=0.5,
                        value=0.2,
                        step=0.1
                    ) 
                
            
            with gr.Row():
                example_btn = gr.Button(t['example_btn'], elem_classes='botaoExemplo')
            
            
        
        with gr.Column(scale=5, elem_classes="mybox"):
            with gr.Tab(t['results']):
                results_text = gr.Markdown(t['results_text'])
            run_btn = gr.Button(t['run_btn'], elem_classes='enviar', variant="primary")
            
                                                                  
    
                    
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