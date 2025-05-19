"""
Main entry point for the AlphaEvolve Pro application.
Orchestrates the different agents and manages the evolutionary loop.
"""
import asyncio
import logging
import sys
import os

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting OpenAlpha_Evolve autonomous algorithmic evolution")
    logger.info(f"Configuration: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")
    logger.info(f"LLM Models: Pro={settings.GEMINI_PRO_MODEL_NAME}, Flash={settings.GEMINI_FLASH_MODEL_NAME}, Eval={settings.GEMINI_EVALUATION_MODEL}")

    # Define the algorithmic task: Generic Shortest Path Problem
    # The input_output_examples remain the same as they define the problem\'s
    # functional contract (what inputs produce what outputs), not the specific algorithm.
    task = TaskDefinition(
        id="generic_shortest_path_problem",
        description=(
            "Given a weighted, directed graph and a starting node, find the shortest distance "
            "from the starting node to all other nodes in the graph. "
            "The graph is represented as a dictionary where keys are node identifiers (e.g., strings or integers), "
            "and values are dictionaries representing outgoing edges. In these inner dictionaries, "
            "keys are neighbor node identifiers and values are the weights (costs) of the edges to those neighbors. "
            "If a node is unreachable from the start node, its distance should be considered infinity. "
            "The function should return a dictionary where keys are node identifiers and values are the "
            "calculated shortest distances from the start node. The start node\'s distance to itself is 0."
        ),
        function_name_to_evolve="solve_shortest_paths", # More generic name
        input_output_examples=[
            {
                "input": [{"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}, "A"],
                "output": {"A": 0, "B": 1, "C": 3, "D": 4}
            },
            {
                "input": [{"A": {"B": 1}, "B": {"A": 2, "C": 5}, "C": {"D": 1}, "D": {}}, "A"],
                "output": {"A": 0, "B": 1, "C": 6, "D": 7}
            },
            { # Disconnected node
                "input": [{"A": {"B": 1}, "B": {}, "C": {"D": 1}, "D": {}}, "A"],
                "output": {"A": 0, "B": 1, "C": float('inf'), "D": float('inf')}
            },
            { # No path to any other node
                "input": [{"A": {}, "B": {"C":1}}, "A"],
                "output": {"A": 0, "B": float('inf'), "C": float('inf')}
            },
            { # Empty graph
                "input": [{}, "A"],
                "output": {"A": 0} # Or could be an error, but let\'s define it as start node 0, others inf if not present
                                 # This interpretation implies nodes not in the graph description but reachable via start_node
                                 # should be handled. Let\'s refine the example to reflect that if a node isn\'t in the graph,
                                 # it shouldn\'t be in output unless it IS the start_node.
                                 # Corrected: output only contains nodes reachable or the start_node.
            },
             { # Empty graph with defined start node (Revisiting based on previous thought)
                "input": [{}, "A"], # Graph is empty, start_node is A
                "output": {"A": 0}  # Only start node is relevant, distance to self is 0
            },
            { # Start node not in graph keys, but is the start
                "input": [{"X": {"Y":1}}, "Z"],
                "output": {"Z": 0, "X": float('inf'), "Y": float('inf')} # Assuming Z is a valid node, but no paths from it in this graph structure
            },
            {
                "input": [
                    {"s": {"u": 10, "x": 5}, "u": {"v": 1, "x": 2}, "v": {"y": 4}, "x": {"u": 3, "v": 9, "y": 2}, "y": {"s": 7, "v": 6}},
                    "s"
                ],
                "output": {"s": 0, "u": 7, "v": 8, "x": 5, "y": 7} # Classic Dijkstra example
            },
            { # Larger graph with more complex paths
                "input": [
                    {
                        "A": {"B": 1, "C": 10},
                        "B": {"D": 2, "E": 5},
                        "C": {"F": 1},
                        "D": {"G": 3},
                        "E": {"G": 1, "H": 7},
                        "F": {"H": 2},
                        "G": {"I": 2},
                        "H": {"I": 1},
                        "I": {}
                    },
                    "A"
                ],
                "output": {"A": 0, "B": 1, "C": 10, "D": 3, "E": 6, "F": 11, "G": 6, "H": 12, "I": 8}
            }
        ],
        allowed_imports=["heapq"],  # Keeping heapq as it\'s generally useful for efficient solutions
        # Optional: Add evaluation criteria or more specific initial code prompts if needed
    )

    # Initialize the TaskManagerAgent with the task definition
    task_manager = TaskManagerAgent(
        task_definition=task
    )

    # Execute the evolutionary process
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
