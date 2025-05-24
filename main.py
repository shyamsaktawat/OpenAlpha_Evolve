"""
Main entry point for the AlphaEvolve Pro application.
Orchestrates the different agents and manages the evolutionary loop.
"""
import asyncio
import logging
import sys
import os
import yaml
import argparse
                                               
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
from config import settings

                   
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)

def load_task_from_yaml(yaml_path: str) -> tuple[list, str, str, str, list]:
    """Load task configuration and test cases from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # Get task configuration
            task_id = data.get('task_id')
            task_description = data.get('task_description')
            function_name = data.get('function_name')
            allowed_imports = data.get('allowed_imports', [])
            
            # Convert test cases from YAML format to input_output_examples format
            input_output_examples = []
            for test_group in data.get('tests', []):
                for test_case in test_group.get('test_cases', []):
                    if 'output' in test_case:
                        input_output_examples.append({
                            'input': test_case['input'],
                            'output': test_case['output']
                        })
                    elif 'validation_func' in test_case:
                        input_output_examples.append({
                            'input': test_case['input'],
                            'validation_func': test_case['validation_func']
                        })
            
            return input_output_examples, task_id, task_description, function_name, allowed_imports
    except Exception as e:
        logger.error(f"Error loading task from YAML: {e}")
        return [], "", "", "", []

async def main():
    parser = argparse.ArgumentParser(description="Run OpenAlpha_Evolve with a specified YAML configuration file.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    yaml_path = args.yaml_path

    logger.info("Starting OpenAlpha_Evolve autonomous algorithmic evolution")
    logger.info(f"Configuration: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")

    # Load task configuration and test cases from YAML file
    test_cases, task_id, task_description, function_name, allowed_imports = load_task_from_yaml(yaml_path)
    
    if not test_cases or not task_id or not task_description or not function_name:
        logger.error("Missing required task configuration in YAML file. Exiting.")
        return

    task = TaskDefinition(
        id=task_id,
        description=task_description,
        function_name_to_evolve=function_name,
        input_output_examples=test_cases,
        allowed_imports=allowed_imports
    )

    task_manager = TaskManagerAgent(
        task_definition=task
    )

    best_programs = await task_manager.execute()

    if best_programs:
        logger.info(f"Evolutionary process completed. Best program(s) found: {len(best_programs)}")
        for i, program in enumerate(best_programs):
            logger.info(f"Final Best Program {i+1} ID: {program.id}")
            logger.info(f"Final Best Program {i+1} Fitness: {program.fitness_scores}")
            logger.info(f"Final Best Program {i+1} Code:\n{program.code}")
    else:
        logger.info("Evolutionary process completed, but no suitable programs were found.")

    logger.info("OpenAlpha_Evolve run finished.")

if __name__ == "__main__":
    asyncio.run(main())
