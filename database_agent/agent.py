import logging
from typing import List, Dict, Any, Optional, Literal
import uuid
import json # Added for JSON operations
import os # Added for file existence check
import asyncio # Added for Lock

from core.interfaces import (
    DatabaseAgentInterface,
    Program,
    BaseAgent,
)
from config import settings # Added to access DATABASE_PATH

logger = logging.getLogger(__name__)

class InMemoryDatabaseAgent(DatabaseAgentInterface, BaseAgent):
    """An in-memory database that persists to a JSON file."""
    def __init__(self):
        super().__init__()
        self._programs: Dict[str, Program] = {}
        self._db_file_path = settings.DATABASE_PATH
        self._lock = asyncio.Lock() # Lock for file operations
        self._load_from_file() # Load existing data on init
        logger.info(f"InMemoryDatabaseAgent initialized. Data persistence: {self._db_file_path}")

    def _load_from_file(self):
        # This is a synchronous load on init. If it needs to be async,
        # it should be called from an async context or __init__ becomes async.
        # For simplicity at startup, keeping it sync.
        if os.path.exists(self._db_file_path):
            try:
                with open(self._db_file_path, 'r') as f:
                    data = json.load(f)
                    for prog_id, prog_data in data.items():
                        # Re-hydrate Program objects
                        # Assuming Program dataclass can be instantiated from dict
                        self._programs[prog_id] = Program(**prog_data) 
                logger.info(f"Loaded {len(self._programs)} programs from {self._db_file_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {self._db_file_path}. Starting with an empty database.")
                self._programs = {}
            except Exception as e:
                logger.error(f"Error loading database from {self._db_file_path}: {e}. Starting with an empty database.")
                self._programs = {}
        else:
            logger.info(f"Database file {self._db_file_path} not found. Starting with an empty database.")

    async def _save_to_file(self):
        async with self._lock:
            try:
                # Serialize Program objects to dictionaries
                data_to_save = {prog_id: prog.__dict__ for prog_id, prog in self._programs.items()}
                with open(self._db_file_path, 'w') as f:
                    json.dump(data_to_save, f, indent=4)
                logger.debug(f"Successfully saved {len(self._programs)} programs to {self._db_file_path}")
            except Exception as e:
                logger.error(f"Error saving database to {self._db_file_path}: {e}")

    async def save_program(self, program: Program) -> None:
        logger.info(f"Saving program: {program.id} (Generation: {program.generation}) to database.")
        async with self._lock:
            if program.id in self._programs:
                logger.warning(f"Program with ID {program.id} already exists. It will be overwritten.")
            self._programs[program.id] = program
        await self._save_to_file() # Persist after every save
        logger.debug(f"Program {program.id} data: {program}")

    async def get_program(self, program_id: str) -> Optional[Program]:
        logger.debug(f"Attempting to retrieve program by ID: {program_id}")
        # No lock needed for read if _programs is mostly read-after-write and writes are locked
        program = self._programs.get(program_id)
        if program:
            logger.info(f"Retrieved program: {program.id}")
        else:
            logger.warning(f"Program with ID: {program_id} not found in database.")
        return program

    async def get_all_programs(self) -> List[Program]:
        logger.debug(f"Retrieving all {len(self._programs)} programs from database.")
        return list(self._programs.values())

    async def get_best_programs(
        self,
        task_id: str, # task_id is not strictly used by InMemoryDB for filtering, but part of interface
        limit: int = 5,
        objective: Literal["correctness", "runtime_ms"] = "correctness",
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> List[Program]:
        logger.info(f"Retrieving best programs (task: {task_id}). Limit: {limit}, Objective: {objective}, Order: {sort_order}")
        if not self._programs:
            logger.info("No programs in database to retrieve 'best' from.")
            return []

        all_progs = list(self._programs.values())

        if objective == "correctness":
            sorted_programs = sorted(all_progs, key=lambda p: p.fitness_scores.get("correctness", 0.0), reverse=(sort_order == "desc"))
        elif objective == "runtime_ms":
            # For runtime_ms: sort_order='asc' (best) means reverse=False (lowest first)
            # sort_order='desc' (worst) means reverse=True (highest first)
            sorted_programs = sorted(all_progs, key=lambda p: p.fitness_scores.get("runtime_ms", float('inf')), reverse=(sort_order == "desc"))
        else:
            logger.warning(f"Unknown objective: {objective}. Defaulting to no specific sort order beyond Program ID.")
            return sorted(all_progs, key=lambda p: p.id)[:limit]

        logger.debug(f"Sorted {len(sorted_programs)} programs. Top 3 (if available): {[p.id for p in sorted_programs[:3]]}")
        return sorted_programs[:limit]

    async def get_programs_by_generation(self, generation: int) -> List[Program]:
        logger.debug(f"Retrieving programs for generation: {generation}")
        generation_programs = [p for p in self._programs.values() if p.generation == generation]
        logger.info(f"Found {len(generation_programs)} programs for generation {generation}.")
        return generation_programs

    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        logger.info(f"Attempting to retrieve {generation_size} programs for next generation for task {task_id}.")
        # Filter by task_id if Program objects have it and it's relevant for this agent
        # For InMemory, usually it holds all programs across tasks unless explicitly designed otherwise.
        # If self._programs can contain programs from multiple tasks, filtering by task_id here is crucial.
        # Assuming for now it holds programs relevant to the current TaskManager context.
        all_relevant_progs = [p for p in self._programs.values() if getattr(p, 'task_id', None) == task_id or task_id is None]
        if not all_relevant_progs:
            logger.warning(f"No programs found for task {task_id} in database to select for next generation.")
            return []

        if len(all_relevant_progs) <= generation_size:
            logger.debug(f"Returning all {len(all_relevant_progs)} programs for task {task_id} as it's <= generation_size {generation_size}.")
            return all_relevant_progs
        
        import random
        selected_programs = random.sample(all_relevant_progs, generation_size)
        logger.info(f"Selected {len(selected_programs)} random programs for task {task_id} for next generation.")
        return selected_programs

    async def count_programs(self) -> int:
        count = len(self._programs)
        logger.debug(f"Total programs in database: {count}")
        return count

    async def clear_database(self) -> None:
        logger.info("Clearing all programs from database.")
        async with self._lock:
            self._programs.clear()
        await self._save_to_file() # Persist the empty state
        logger.info("Database cleared.")

    async def execute(self, *args, **kwargs) -> Any:
        logger.warning("InMemoryDatabaseAgent.execute() called, but this agent uses specific methods for DB operations.")
        raise NotImplementedError("InMemoryDatabaseAgent does not have a generic execute. Use specific methods like save_program, get_program etc.")

                                      
if __name__ == "__main__":
    import asyncio                                                    
    async def test_db():
        logging.basicConfig(level=logging.DEBUG)
        
        # Mock settings for testing
        class MockSettings:
            DATABASE_PATH = "test_inmemory_agent.json"
        global settings
        settings = MockSettings()

        # Clean up previous test file
        if os.path.exists(settings.DATABASE_PATH):
            os.remove(settings.DATABASE_PATH)

        db = InMemoryDatabaseAgent()

        prog1_data = {"id":"prog_001", "code":"print('hello')", "generation":0, "fitness_scores":{"correctness": 0.8, "runtime_ms": 100}, "task_id": "test_task"}
        prog2_data = {"id":"prog_002", "code":"print('world')", "generation":0, "fitness_scores":{"correctness": 0.9, "runtime_ms": 50}, "task_id": "test_task"}
        prog3_data = {"id":"prog_003", "code":"print('test')", "generation":1, "fitness_scores":{"correctness": 0.85, "runtime_ms": 70}, "task_id": "test_task"}

        prog1 = Program(**prog1_data)
        prog2 = Program(**prog2_data)
        prog3 = Program(**prog3_data)

        await db.save_program(prog1)
        await db.save_program(prog2)
        await db.save_program(prog3)

        retrieved_prog = await db.get_program("prog_001")
        assert retrieved_prog is not None and retrieved_prog.code == "print('hello')"
        assert retrieved_prog.task_id == "test_task"

        all_programs = await db.get_all_programs()
        assert len(all_programs) == 3

        # Test loading from file by creating a new instance
        db2 = InMemoryDatabaseAgent()
        assert await db2.count_programs() == 3
        retrieved_prog2 = await db2.get_program("prog_002")
        assert retrieved_prog2 is not None and retrieved_prog2.fitness_scores.get("correctness") == 0.9

        best_correctness = await db.get_best_programs(task_id="test_task", limit=2, objective="correctness", sort_order="desc")
        print(f"Best by correctness (desc): {[p.id for p in best_correctness]}")
        assert len(best_correctness) == 2
        assert best_correctness[0].id == "prog_002"      
        assert best_correctness[1].id == "prog_003"       

        best_runtime_asc = await db.get_best_programs(task_id="test_task", limit=2, objective="runtime_ms", sort_order="asc")
        print(f"Best by runtime (asc): {[p.id for p in best_runtime_asc]}")
        assert len(best_runtime_asc) == 2
        assert best_runtime_asc[0].id == "prog_002"
        # Corrected assertion for runtime, prog3 (70ms) is better than prog1 (100ms) when ascending
        assert best_runtime_asc[1].id == "prog_003"
        
        next_gen_task_programs = await db.get_programs_for_next_generation(task_id="test_task", generation_size=2)
        assert len(next_gen_task_programs) == 2
        for p in next_gen_task_programs:
            assert p.task_id == "test_task"

        await db.clear_database()
        assert await db.count_programs() == 0
        assert not os.path.exists(settings.DATABASE_PATH) or os.path.getsize(settings.DATABASE_PATH) < 5 # empty json is like {} or []
        print("InMemoryDatabaseAgent with JSON persistence tests passed.")

        # Cleanup test file
        if os.path.exists(settings.DATABASE_PATH):
            os.remove(settings.DATABASE_PATH)

    asyncio.run(test_db()) 