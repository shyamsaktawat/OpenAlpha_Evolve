                 
import logging
from typing import List, Dict, Any, Optional, Literal
import uuid

from core.interfaces import (
    DatabaseAgentInterface,
    Program,
    BaseAgent,
)
                                                               

logger = logging.getLogger(__name__)

class InMemoryDatabaseAgent(DatabaseAgentInterface, BaseAgent):
    """An in-memory database for storing and retrieving programs."""
    def __init__(self):
        super().__init__()
        self._programs: Dict[str, Program] = {}
        logger.info("InMemoryDatabaseAgent initialized.")

    async def save_program(self, program: Program) -> None:
        logger.info(f"Saving program: {program.id} (Generation: {program.generation}) to in-memory database.")
        if program.id in self._programs:
            logger.warning(f"Program with ID {program.id} already exists (expected during evolution/migration). It will be overwritten.")
        self._programs[program.id] = program
        logger.debug(f"Program {program.id} data: {program}")

    async def get_program(self, program_id: str) -> Optional[Program]:
        logger.debug(f"Attempting to retrieve program by ID: {program_id}")
        program = self._programs.get(program_id)
        if program:
            logger.info(f"Retrieved program: {program.id}")
        else:
            logger.warning(f"Program with ID: {program_id} not found in database.")
        return program

    async def get_all_programs(self) -> List[Program]:
        logger.debug(f"Retrieving all {len(self._programs)} programs from in-memory database.")
        return list(self._programs.values())

    async def get_best_programs(
        self,
        task_id: str,
        limit: int = 5,
        objective: Literal["correctness", "runtime_ms"] = "correctness",
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> List[Program]:
        logger.info(f"Retrieving best programs for task {task_id}. Limit: {limit}, Objective: {objective}, Order: {sort_order}")
        if not self._programs:
            logger.info("No programs in database to retrieve 'best' from.")
            return []

        all_progs = list(self._programs.values())
                                                                              
                                                                                                                
                                                                                                          
        relevant_progs = all_progs

        def sort_key(p: Program):
            if objective == "correctness":
                return p.fitness_scores.get("correctness", -1.0)
            elif objective == "runtime_ms":
                val = p.fitness_scores.get("runtime_ms", float('inf'))
                return -val if sort_order == "desc" else val 
            return 0                                       

                                                                          
        if objective == "correctness":                               
            effective_reverse = (sort_order == "desc")
        elif objective == "runtime_ms":                             
            effective_reverse = (sort_order == "asc")                                                 
        else:
            effective_reverse = False          
        
                                                                   
        if objective == "runtime_ms":
                                                                            
                                                                           
            sorted_programs = sorted(relevant_progs, key=lambda p: p.fitness_scores.get("runtime_ms", float('inf')), reverse=(sort_order == "desc"))
        elif objective == "correctness":
                                                               
                                                                
            sorted_programs = sorted(relevant_progs, key=lambda p: p.fitness_scores.get("correctness", -1.0), reverse=(sort_order == "desc"))
        else:
                                                           
            sorted_programs = sorted(relevant_progs, key=sort_key, reverse=effective_reverse)

        logger.debug(f"Sorted {len(sorted_programs)} programs. Top 3 (if available): {[p.id for p in sorted_programs[:3]]}")
        return sorted_programs[:limit]

    async def get_programs_by_generation(self, generation: int) -> List[Program]:
        logger.debug(f"Retrieving programs for generation: {generation}")
        generation_programs = [p for p in self._programs.values() if p.generation == generation]
        logger.info(f"Found {len(generation_programs)} programs for generation {generation}.")
        return generation_programs

    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        logger.info(f"Attempting to retrieve {generation_size} programs for next generation for task {task_id}.")
        all_progs = list(self._programs.values())
        if not all_progs:
            logger.warning("No programs in database to select for next generation.")
            return []

        if len(all_progs) <= generation_size:
            logger.debug(f"Returning all {len(all_progs)} programs as it's less than or equal to generation_size {generation_size}.")
            return all_progs
        
        import random
        selected_programs = random.sample(all_progs, generation_size)
        logger.info(f"Selected {len(selected_programs)} random programs for next generation.")
        return selected_programs

    async def count_programs(self) -> int:
        count = len(self._programs)
        logger.debug(f"Total programs in database: {count}")
        return count

    async def clear_database(self) -> None:
        logger.info("Clearing all programs from in-memory database.")
        self._programs.clear()
        logger.info("In-memory database cleared.")

    async def execute(self, *args, **kwargs) -> Any:
        logger.warning("InMemoryDatabaseAgent.execute() called, but this agent uses specific methods for DB operations.")
        raise NotImplementedError("InMemoryDatabaseAgent does not have a generic execute. Use specific methods like save_program, get_program etc.")

                                      
if __name__ == "__main__":
    import asyncio                                                    
    async def test_db():
        logging.basicConfig(level=logging.DEBUG)
        db = InMemoryDatabaseAgent()

        prog1 = Program(id="prog_001", code="print('hello')", generation=0, fitness_scores={"correctness_score": 0.8, "runtime_ms": 100})
        prog2 = Program(id="prog_002", code="print('world')", generation=0, fitness_scores={"correctness_score": 0.9, "runtime_ms": 50})
        prog3 = Program(id="prog_003", code="print('test')", generation=1, fitness_scores={"correctness_score": 0.85, "runtime_ms": 70})

        await db.save_program(prog1)
        await db.save_program(prog2)
        await db.save_program(prog3)

        retrieved_prog = await db.get_program("prog_001")
        assert retrieved_prog is not None and retrieved_prog.code == "print('hello')"

        all_programs = await db.get_all_programs()
        assert len(all_programs) == 3

        best_correctness = await db.get_best_programs(task_id="test_task", limit=2, objective="correctness", sort_order="desc")
        print(f"Best by correctness (desc): {[p.id for p in best_correctness]}")
        assert len(best_correctness) == 2
        assert best_correctness[0].id == "prog_002"      
        assert best_correctness[1].id == "prog_003"       

        best_runtime_asc = await db.get_best_programs(task_id="test_task", limit=2, objective="runtime_ms", sort_order="asc")
        print(f"Best by runtime (asc): {[p.id for p in best_runtime_asc]}")
        assert len(best_runtime_asc) == 2
        assert best_runtime_asc[0].id == "prog_002"       
        assert best_runtime_asc[1].id == "prog_003"       
        
        best_runtime_desc = await db.get_best_programs(task_id="test_task", limit=2, objective="runtime_ms", sort_order="desc")
        print(f"Best by runtime (desc): {[p.id for p in best_runtime_desc]}")
        assert len(best_runtime_desc) == 2
        assert best_runtime_desc[0].id == "prog_001"        
        assert best_runtime_desc[1].id == "prog_003"       

                                               
        next_gen_programs = await db.get_programs_for_next_generation(task_id="test_task", generation_size=2)
        print(f"Next gen programs (size 2): {[p.id for p in next_gen_programs]}")
        assert len(next_gen_programs) == 2

        next_gen_programs_all = await db.get_programs_for_next_generation(task_id="test_task", generation_size=5)
        print(f"Next gen programs (size 5, all): {[p.id for p in next_gen_programs_all]}")
        assert len(next_gen_programs_all) == 3                              

        gen0_programs = await db.get_programs_by_generation(0)
        assert len(gen0_programs) == 2

        assert await db.count_programs() == 3
        
        await db.clear_database()
        assert await db.count_programs() == 0
        print("InMemoryDatabaseAgent tests passed.")

    asyncio.run(test_db()) 