import random
import logging
from typing import List, Dict, Any, Optional

from core.interfaces import SelectionControllerInterface, Program, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

class SelectionControllerAgent(SelectionControllerInterface, BaseAgent):
    def __init__(self):
        super().__init__()
        self.elitism_count = settings.ELITISM_COUNT
        logger.info(f"SelectionControllerAgent initialized with elitism_count: {self.elitism_count}")

    def select_parents(self, population: List[Program], num_parents: int) -> List[Program]:
        logger.info(f"Starting parent selection. Population size: {len(population)}, Number of parents to select: {num_parents}")
        if not population:
            logger.warning("Parent selection called with empty population. Returning empty list.")
            return []
        if num_parents == 0:
            logger.info("Number of parents to select is 0. Returning empty list.")
            return []
        if num_parents > len(population):
            logger.warning(f"Requested {num_parents} parents, but population size is only {len(population)}. Selecting all individuals as parents.")
            return list(population) # Return a copy

        sorted_population = sorted(
            population,
            key=lambda p: (
                p.fitness_scores.get("correctness", 0.0),
                -p.fitness_scores.get("runtime_ms", float('inf')) # Negative for ascending sort on runtime
            ),
            reverse=True  # Higher correctness is better
        )
        logger.debug(f"Population sorted for parent selection. Top 3 (if available): {[p.id for p in sorted_population[:3]]}")

        parents = []

        elite_candidates = []
        seen_ids_for_elitism = set()
        for program in sorted_population:
            if len(elite_candidates) < self.elitism_count:
                if program.id not in seen_ids_for_elitism: # Ensure uniqueness if programs can have same fitness
                    elite_candidates.append(program)
                    seen_ids_for_elitism.add(program.id)
            else:
                break
        parents.extend(elite_candidates)
        logger.info(f"Selected {len(elite_candidates)} elite parents: {[p.id for p in elite_candidates]}")

        remaining_slots = num_parents - len(parents)
        if remaining_slots <= 0:
            logger.info("Elitism filled all parent slots or no more parents needed.")
            return parents

        roulette_candidates = [p for p in sorted_population if p.id not in seen_ids_for_elitism]
        if not roulette_candidates:
            logger.warning("No candidates left for roulette selection after elitism. Returning current parents.")
            return parents

        total_fitness = sum(p.fitness_scores.get("correctness", 0.0) + 0.0001 for p in roulette_candidates)
        logger.debug(f"Total fitness for roulette wheel selection (among {len(roulette_candidates)} candidates): {total_fitness:.4f}")

        if total_fitness <= 0.0001 * len(roulette_candidates): # All effectively zero
            logger.warning("All roulette candidates have near-zero fitness. Selecting randomly from them.")
            num_to_select_randomly = min(remaining_slots, len(roulette_candidates))
            random_parents = random.sample(roulette_candidates, num_to_select_randomly)
            parents.extend(random_parents)
            logger.info(f"Selected {len(random_parents)} parents randomly due to zero total fitness: {[p.id for p in random_parents]}")
        else:
            for _ in range(remaining_slots):
                if not roulette_candidates: break # Should not happen if logic is correct
                pick = random.uniform(0, total_fitness)
                current_sum = 0
                chosen_parent = None
                for program in roulette_candidates:
                    current_sum += (program.fitness_scores.get("correctness", 0.0) + 0.0001)
                    if current_sum >= pick:
                        chosen_parent = program
                        break
                if chosen_parent:
                    parents.append(chosen_parent)
                    logger.debug(f"Selected parent via roulette: {chosen_parent.id} (Fitness: {chosen_parent.fitness_scores.get('correctness')})")
                else:
                    if roulette_candidates: # Should always be true here
                        fallback_parent = random.choice(roulette_candidates)
                        parents.append(fallback_parent)
                        logger.warning(f"Roulette selection resulted in no choice (edge case), picked randomly: {fallback_parent.id}")
                    else:
                        logger.warning("Roulette selection ran out of candidates (edge case).")
                        break 

        logger.info(f"Total parents selected: {len(parents)}. IDs: {[p.id for p in parents]}")
        return parents

    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int) -> List[Program]:
        logger.info(f"Starting survivor selection. Current pop: {len(current_population)}, Offspring pop: {len(offspring_population)}, Target pop size: {population_size}")
        combined_population = current_population + offspring_population
        logger.debug(f"Combined population size for survivor selection: {len(combined_population)}")

        if not combined_population:
            logger.warning("Survivor selection called with empty combined population. Returning empty list.")
            return []

        sorted_combined = sorted(
            combined_population,
            key=lambda p: (
                p.fitness_scores.get("correctness", 0.0),
                -p.fitness_scores.get("runtime_ms", float('inf')),
                -p.generation # Favor newer generations in case of exact fitness tie
            ),
            reverse=True
        )
        logger.debug(f"Combined population sorted for survivor selection. Top 3 (if available): {[p.id for p in sorted_combined[:3]]}")

        survivors = []
        seen_program_ids = set()
        for program in sorted_combined:
            if len(survivors) < population_size:
                if program.id not in seen_program_ids:
                    survivors.append(program)
                    seen_program_ids.add(program.id)
            else:
                break
        
        logger.info(f"Selected {len(survivors)} survivors. IDs: {[p.id for p in survivors]}")
        return survivors

    async def execute(self, action: str, **kwargs) -> Any:
        if action == "select_parents":
            return self.select_parents(kwargs['population'], kwargs['num_parents'])
        elif action == "select_survivors":
            return self.select_survivors(kwargs['current_population'], kwargs['offspring_population'], kwargs['population_size'])
        else:
            raise ValueError(f"Unknown action: {action}")

if __name__ == '__main__':
    import uuid
    logging.basicConfig(level=logging.DEBUG)
    selector = SelectionControllerAgent()

    programs = [
        Program(id=str(uuid.uuid4()), code="c1", fitness_scores={"correctness": 0.9, "runtime_ms": 100}, status="evaluated"),
        Program(id=str(uuid.uuid4()), code="c2", fitness_scores={"correctness": 1.0, "runtime_ms": 50}, status="evaluated"),
        Program(id=str(uuid.uuid4()), code="c3", fitness_scores={"correctness": 0.7, "runtime_ms": 200}, status="evaluated"),
        Program(id=str(uuid.uuid4()), code="c4", fitness_scores={"correctness": 1.0, "runtime_ms": 60}, status="evaluated"), # Duplicate high correctness
        Program(id=str(uuid.uuid4()), code="c5", fitness_scores={"correctness": 0.5}, status="evaluated"), # Missing runtime
        Program(id=str(uuid.uuid4()), code="c6", status="unevaluated"), # Unevaluated
    ]

    print("--- Testing Parent Selection ---")
    parents = selector.select_parents(programs, num_parents=3)
    for p in parents:
        print(f"Selected Parent: {p.id}, Correctness: {p.fitness_scores.get('correctness')}, Runtime: {p.fitness_scores.get('runtime_ms')}")

    print("\n--- Testing Survivor Selection ---")
    current_pop = programs[:2] # p1, p2
    offspring_pop = [
        Program(id=str(uuid.uuid4()), code="off1", fitness_scores={"correctness": 1.0, "runtime_ms": 40}, status="evaluated"), # Better than p2
        Program(id=str(uuid.uuid4()), code="off2", fitness_scores={"correctness": 0.6, "runtime_ms": 10}, status="evaluated"),
    ]
    survivors = selector.select_survivors(current_pop, offspring_pop, population_size=2)
    for s in survivors:
        print(f"Survivor: {s.id}, Correctness: {s.fitness_scores.get('correctness')}, Runtime: {s.fitness_scores.get('runtime_ms')}") 