from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import time

@dataclass
class Program:
    id: str
    code: str
    fitness_scores: Dict[str, float] = field(default_factory=dict)                                                 
    generation: int = 0
    parent_id: Optional[str] = None
    island_id: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    status: str = "unevaluated"
    created_at: float = field(default_factory=lambda: time.time())  # Track program age
    task_id: Optional[str] = None

@dataclass
class TaskDefinition:
    id: str
    description: str                                              
    function_name_to_evolve: Optional[str] = None  # Can be used if evolving a single function
    target_file_path: Optional[str] = None # Path to the file containing code to be evolved
    evolve_blocks: Optional[List[Dict[str, Any]]] = None # Defines specific blocks within the target_file_path to evolve
                                                        # e.g., [{'block_id': 'optimizer_logic', 'start_marker': '# EVOLVE-BLOCK-START optimizer', 'end_marker': '# EVOLVE-BLOCK-END optimizer'}]
    input_output_examples: Optional[List[Dict[str, Any]]] = None                                                    
    evaluation_criteria: Optional[Dict[str, Any]] = None                                                            
    initial_code_prompt: Optional[str] = "Provide an initial Python solution for the following problem:"
    allowed_imports: Optional[List[str]] = None
    tests: Optional[List[Dict[str, Any]]] = None # List of test groups. Each group is a dict, can include 'name', 'description', 'level' (for cascade), and 'test_cases'.
    expert_knowledge: Optional[str] = None # Relevant expert knowledge, equations, or snippets

class BaseAgent(ABC):
    """Base class for all agents."""
    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Main execution method for an agent."""
        pass

class TaskManagerInterface(BaseAgent):
    @abstractmethod
    async def manage_evolutionary_cycle(self):
        pass

class PromptDesignerInterface(BaseAgent):
    @abstractmethod
    def design_initial_prompt(self, task: TaskDefinition) -> str:
        pass

    @abstractmethod
    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program, evaluation_feedback: Optional[Dict] = None) -> str:
        pass

    @abstractmethod
    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_info: Dict) -> str:
        pass

class CodeGeneratorInterface(BaseAgent):
    @abstractmethod
    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = 0.7, output_format: str = "code") -> str:
        pass

class EvaluatorAgentInterface(BaseAgent):
    @abstractmethod
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        pass

class DatabaseAgentInterface(BaseAgent):
    @abstractmethod
    async def save_program(self, program: Program):
        pass

    @abstractmethod
    async def get_program(self, program_id: str) -> Optional[Program]:
        pass

    @abstractmethod
    async def get_best_programs(self, task_id: str, limit: int = 10, objective: Optional[str] = None) -> List[Program]:
        pass
    
    @abstractmethod
    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        pass

class SelectionControllerInterface(BaseAgent):
    @abstractmethod
    def select_parents(self, evaluated_programs: List[Program], num_parents: int) -> List[Program]:
        pass

    @abstractmethod
    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int) -> List[Program]:
        pass

    @abstractmethod
    def initialize_islands(self, initial_programs: List[Program]) -> None:
        pass

class RLFineTunerInterface(BaseAgent):
    @abstractmethod
    async def update_policy(self, experience_data: List[Dict]):
        pass

class MonitoringAgentInterface(BaseAgent):
    @abstractmethod
    async def log_metrics(self, metrics: Dict):
        pass

    @abstractmethod
    async def report_status(self):
        pass

                                                                      