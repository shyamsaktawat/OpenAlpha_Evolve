"""
Gradio web interface for OpenAlpha_Evolve.
MODERNIZED UX/UI - OLLAMA INTEGRATION - ENHANCED FUNCTIONALITY
"""
import gradio as gr
import asyncio
import json
import os
import sys
import time
import logging
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# --- Project Root Setup ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

from core.interfaces import TaskDefinition
from task_manager.agent import TaskManagerAgent
from config import settings

# --- Logging Setup ---
class UILoggingHandler(logging.Handler):
    """A handler that captures logs and can be displayed in a Gradio Textbox."""
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

# --- Global Handlers and Loggers ---
ui_handler = UILoggingHandler()
ui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set root logger level
root_logger.addHandler(ui_handler) # Add our custom UI handler

# Console handler for backend visibility
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

# Set specific log levels for noisy modules if needed
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- State Management ---
current_results = []

# --- Ollama Integration ---
def get_ollama_models():
    """
    Fetches the list of installed Ollama models.
    Returns a list of model names and an error message if any.
    """
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        # Skip header and parse model names
        models = [line.split()[0] for line in lines[1:]]
        if not models:
            return [], "Nenhum modelo Ollama encontrado. Verifique sua instalação."
        return models, "Modelos carregados com sucesso!"
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(f"Ollama command failed: {e}")
        return [], "Erro: O comando 'ollama' não foi encontrado. Certifique-se de que o Ollama está instalado e no seu PATH."
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching Ollama models: {e}")
        return [], f"Um erro inesperado ocorreu: {e}"

# --- Core Evolution Logic ---
async def run_evolution(
    # Inputs from UI
    task_id, description, function_name, examples_json, allowed_imports_text,
    population_size, generations, num_islands, migration_frequency, migration_rate,
    ollama_model,
    # Gradio progress tracker
    progress=gr.Progress(track_tqdm=True)
):
    """
    Run the evolutionary process with the given parameters and stream logs.
    """
    ui_handler.clear()
    global current_results
    current_results = []
    
    yield "Inicializando a evolução...", "", gr.Button("🚀 Executando...", interactive=False)

    try:
        # --- 1. Validate Inputs ---
        try:
            examples = json.loads(examples_json)
            if not isinstance(examples, list) or not all(isinstance(ex, dict) and "input" in ex and "output" in ex for ex in examples):
                raise ValueError("Exemplos devem ser uma lista de objetos JSON com chaves 'input' e 'output'.")
        except (json.JSONDecodeError, ValueError) as e:
            yield str(e), ui_handler.get_logs(), gr.Button("🚀 Run Evolution", interactive=True)
            return

        allowed_imports = [imp.strip() for imp in allowed_imports_text.split(",") if imp.strip()]

        # --- 2. Configure Settings ---
        settings.POPULATION_SIZE = int(population_size)
        settings.GENERATIONS = int(generations)
        settings.NUM_ISLANDS = int(num_islands)
        settings.MIGRATION_FREQUENCY = int(migration_frequency)
        settings.MIGRATION_RATE = float(migration_rate)
        
        # KEY CHANGE: Set the model from the UI for the agent to use
        settings.LITELLM_DEFAULT_MODEL = f"ollama/{ollama_model}"
        settings.LITELLM_DEFAULT_BASE_URL = "http://localhost:11434" # Standard Ollama API endpoint

        logger.info(f"Evolução configurada com o modelo: {settings.LITELLM_DEFAULT_MODEL}")
        yield "Configurações aplicadas. Preparando a tarefa...", ui_handler.get_logs(), gr.Button("🚀 Executando...", interactive=False)

        # --- 3. Define Task ---
        task = TaskDefinition(
            id=task_id,
            description=description,
            function_name_to_evolve=function_name,
            input_output_examples=examples,
            allowed_imports=allowed_imports
        )

        # --- 4. Run Evolution & Stream Logs ---
        task_manager = TaskManagerAgent(task_definition=task)

        # Use a separate task to run the evolution so the UI remains responsive
        evolution_task = asyncio.create_task(task_manager.execute())

        while not evolution_task.done():
            progress(0, desc="Evoluindo... Verifique a aba de Logs para detalhes.")
            yield "Processo em andamento...", ui_handler.get_logs(), gr.Button("🚀 Executando...", interactive=False)
            await asyncio.sleep(1) # Update interval for logs

        best_programs = await evolution_task
        
        # --- 5. Process and Display Results ---
        current_results = best_programs if best_programs else []
        
        if best_programs:
            result_text = f"✅ **Evolução concluída com sucesso!** Encontrada(s) {len(best_programs)} solução(ões).\n\n"
            for i, program in enumerate(best_programs):
                result_text += f"### Solução {i+1} (Fitness: {program.fitness_scores})\n"
                result_text += f"**ID:** `{program.id}` | **Geração:** {program.generation} | **Ilha:** {program.island_id}\n\n"
                result_text += f"```python\n{program.code}\n```\n\n---\n"
            final_log = ui_handler.get_logs()
            yield result_text, final_log, gr.Button("🚀 Run Evolution", interactive=True)
        else:
            yield "❌ **Evolução concluída, mas nenhuma solução viável foi encontrada.**", ui_handler.get_logs(), gr.Button("🚀 Run Evolution", interactive=True)

    except Exception as e:
        import traceback
        error_msg = f"Ocorreu um erro crítico durante a evolução: {str(e)}\n\n{traceback.format_exc()}"
        logger.error(error_msg)
        yield error_msg, ui_handler.get_logs(), gr.Button("🚀 Run Evolution", interactive=True)


# --- UI Examples ---
FIB_EXAMPLES = '''[
    {"input": [0], "output": 0},
    {"input": [1], "output": 1},
    {"input": [5], "output": 5},
    {"input": [10], "output": 55}
]'''

SORT_EXAMPLES = '''[
    {"input": [[3, 1, 4, 1, 5, 9, 2, 6]], "output": [1, 1, 2, 3, 4, 5, 6, 9]},
    {"input": [[], "output": []},
    {"input": [[5, -1, 0, 10]], "output": [-1, 0, 5, 10]}
]'''

def set_fib_example():
    return "fibonacci_task", "Escreva uma função em Python que calcula o n-ésimo número de Fibonacci (base 0).", "fibonacci", FIB_EXAMPLES, ""

def set_sort_example():
    return "sort_list_task", "Escreva uma função em Python que ordena uma lista de inteiros em ordem crescente.", "sort_list", SORT_EXAMPLES, " "


# --- Build Gradio Interface ---
initial_models, initial_status = get_ollama_models()

with gr.Blocks(theme=gr.themes.Soft(), title="🧬 OpenAlpha_Evolve+") as demo:
    gr.Markdown("# 🧬 OpenAlpha_Evolve+")
    gr.Markdown("Uma interface aprimorada para evolução autônoma de algoritmos com integração nativa ao Ollama.")

    with gr.Row():
        # --- LEFT COLUMN: Configuration ---
        with gr.Column(scale=2):
            gr.Markdown("### 🧠 Configuração do Modelo")
            with gr.Group():
                with gr.Row():
                    ollama_model_dropdown = gr.Dropdown(
                        label="Modelo Ollama",
                        choices=initial_models,
                        value=initial_models[0] if initial_models else None,
                        info="Selecione o modelo Ollama local para usar na geração de código.",
                        interactive=bool(initial_models)
                    )
                    refresh_models_btn = gr.Button("🔄")
                ollama_status_text = gr.Markdown(initial_status)

            gr.Markdown("### 📝 Definição da Tarefa")
            with gr.Group():
                task_id = gr.Textbox(label="ID da Tarefa", value="fibonacci_task", info="Um identificador único para a tarefa.")
                description = gr.Textbox(label="Descrição da Tarefa", lines=3, value="Escreva uma função em Python que calcula o n-ésimo número de Fibonacci (base 0).", info="Descreva o problema que a função deve resolver.")
                function_name = gr.Textbox(label="Nome da Função a ser Evoluída", value="fibonacci", info="O nome da função Python que será gerada.")
                examples_json = gr.Code(label="Exemplos de Entrada/Saída (JSON)", language="json", value=FIB_EXAMPLES, lines=8)
                allowed_imports = gr.Textbox(label="Imports Permitidos", placeholder="ex: math, random", info="Liste as bibliotecas Python permitidas, separadas por vírgula.")

            with gr.Accordion("⚙️ Parâmetros Avançados de Evolução", open=False):
                with gr.Group():
                    population_size = gr.Slider(label="Tamanho da População", minimum=10, maximum=200, value=20, step=10, info="Número de programas em cada geração.")
                    generations = gr.Slider(label="Gerações", minimum=2, maximum=100, value=10, step=1, info="Número de ciclos de evolução a serem executados.")
                    num_islands = gr.Slider(label="Número de Ilhas", minimum=1, maximum=10, value=3, step=1, info="Divide a população para promover diversidade.")
                    migration_frequency = gr.Slider(label="Frequência de Migração", minimum=1, maximum=20, value=5, step=1, info="A cada quantas gerações os programas migram entre ilhas.")
                    migration_rate = gr.Slider(label="Taxa de Migração", minimum=0.05, maximum=0.5, value=0.1, step=0.05, info="Percentual da população que migra.")
            
            with gr.Accordion("📘 Presets de Exemplo", open=False):
                with gr.Row():
                    fib_btn = gr.Button("Fibonacci")
                    sort_btn = gr.Button("Ordenação")

            run_btn = gr.Button("🚀 Run Evolution", variant="primary")

        # --- RIGHT COLUMN: Results & Logs ---
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("📄 Resultados", id="results_tab"):
                    results_text = gr.Markdown("Os resultados da evolução aparecerão aqui...")
                with gr.TabItem("📜 Logs", id="logs_tab"):
                    logs_output = gr.Textbox(label="Log da Execução", lines=25, max_lines=50, interactive=False, autoscroll=True)

    # --- Event Handlers ---
    def refresh_models_and_update_dropdown():
        models, status = get_ollama_models()
        return gr.Dropdown(choices=models, value=models[0] if models else None, interactive=bool(models)), status

    refresh_models_btn.click(
        refresh_models_and_update_dropdown,
        outputs=[ollama_model_dropdown, ollama_status_text]
    )
    
    fib_btn.click(set_fib_example, outputs=[task_id, description, function_name, examples_json, allowed_imports])
    sort_btn.click(set_sort_example, outputs=[task_id, description, function_name, examples_json, allowed_imports])

    run_event = run_btn.click(
        fn=run_evolution,
        inputs=[
            task_id, description, function_name, examples_json, allowed_imports,
            population_size, generations, num_islands, migration_frequency, migration_rate,
            ollama_model_dropdown
        ],
        outputs=[results_text, logs_output, run_btn]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)