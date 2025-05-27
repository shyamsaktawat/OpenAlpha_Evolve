import random
import logging
from typing import List, Dict, Any, Optional

from core.interfaces import SelectionControllerInterface, Program, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

class Island:
    def __init__(self, island_id: int, initial_programs: Optional[List[Program]] = None):
        self.island_id = island_id
        self.programs = initial_programs or []
        self.generation = 0  # Island's internal generation counter
        self.best_fitness = 0.0
        self.last_improvement_generation = 0
        
        if settings.DEBUG:
            logger.debug(f"Initializing Island {island_id} with {len(self.programs)} programs")
        
        for program in self.programs:
            program.island_id = island_id
            # If it's a single program being re-seeded (likely from migration), it retains its original generation.
            # If it's part of a larger initial set for a brand new island, and its generation is uninitialized (e.g. None or 0 by default),
            # then assign the island's starting generation.
            if len(self.programs) > 1 and (program.generation is None or program.generation == 0):
                program.generation = self.generation  # Island's current gen (0 for new)
                if settings.DEBUG:
                    logger.debug(f"Set generation for program {program.id} to {self.generation}")

    def get_best_program(self) -> Optional[Program]:
        if not self.programs:
            return None
        # Sort by correctness (higher is better), runtime (lower is better), generation (lower/older is better), and creation time (older is better)
        best_program = max(
            self.programs,
            key=lambda p: (
                p.fitness_scores.get("correctness", 0.0),  # Higher correctness preferred
                -p.fitness_scores.get("runtime_ms", float('inf')),  # Lower runtime preferred
                -p.generation,  # Older generation preferred
                -p.created_at  # Older creation time preferred as tiebreaker
            )
        )
        if settings.DEBUG:
            logger.debug(f"Island {self.island_id} best program: ID={best_program.id}, "
                        f"Correctness={best_program.fitness_scores.get('correctness')}, "
                        f"Runtime={best_program.fitness_scores.get('runtime_ms')}, "
                        f"Generation={best_program.generation}")
        return best_program

    def update_metrics(self):
        best_program = self.get_best_program()
        if best_program:
            current_best = best_program.fitness_scores.get("correctness", 0.0)
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.last_improvement_generation = self.generation
                if settings.DEBUG:
                    logger.debug(f"Island {self.island_id} new best fitness: {self.best_fitness} "
                               f"at generation {self.generation}")
        self.generation += 1
        if settings.DEBUG:
            logger.debug(f"Island {self.island_id} generation incremented to {self.generation}")

class SelectionControllerAgent(SelectionControllerInterface, BaseAgent):
    def __init__(self):
        super().__init__()
        self.elitism_count = settings.ELITISM_COUNT
        self.num_islands = settings.NUM_ISLANDS
        self.migration_interval = settings.MIGRATION_INTERVAL
        self.islands: Dict[int, Island] = {}
        self.current_generation = 0
        logger.info(f"SelectionControllerAgent initialized with {self.num_islands} islands and elitism_count: {self.elitism_count}")

    def initialize_islands(self, initial_programs: List[Program]) -> None:
        """Initialize islands with the initial population."""
        programs_per_island = len(initial_programs) // self.num_islands
        if settings.DEBUG:
            logger.debug(f"Initializing {self.num_islands} islands with {programs_per_island} programs each")
        
        for i in range(self.num_islands):
            start_idx = i * programs_per_island
            end_idx = start_idx + programs_per_island if i < self.num_islands - 1 else len(initial_programs)
            island_programs = initial_programs[start_idx:end_idx]
            self.islands[i] = Island(i, island_programs)
            if settings.DEBUG:
                logger.debug(f"Initialized Island {i} with {len(island_programs)} programs")

    def select_parents(self, population: List[Program], num_parents: int) -> List[Program]:
        # The 'population' argument here is the global population, but selection is per-island driven.
        logger.debug(f"Starting parent selection. Global population size: {len(population)}, Num parents to select: {num_parents}")
        
        if num_parents == 0:
            logger.info("Number of parents to select is 0. Returning empty list.")
            return []

        all_potential_parents: List[Program] = []
        parents_per_island = max(1, num_parents // self.num_islands) # Ensure at least 1 parent per island if possible
        remaining_parents_to_select = num_parents

        # Prioritize elites from all islands first
        all_elites = []
        for island_id, island in self.islands.items():
            if not island.programs:
                logger.warning(f"Island {island_id} is empty. Skipping for elite selection.")
                continue

            sorted_island_programs = sorted(
                island.programs,
                key=lambda p: (p.fitness_scores.get("correctness", 0.0), -p.fitness_scores.get("runtime_ms", float('inf')), -p.generation),
                reverse=True
            )
            for i in range(min(len(sorted_island_programs), self.elitism_count)):
                 all_elites.append(sorted_island_programs[i])
        
        # Deduplicate elites (in case of migration having same elite in multiple islands)
        unique_elites = []
        seen_elite_ids = set()
        for elite in sorted(all_elites, key=lambda p: (p.fitness_scores.get("correctness", 0.0), -p.fitness_scores.get("runtime_ms", float('inf')), -p.generation), reverse=True):
            if elite.id not in seen_elite_ids:
                unique_elites.append(elite)
                seen_elite_ids.add(elite.id)
        
        # Add top N unique elites directly to parents
        selected_parents = unique_elites[:min(num_parents, len(unique_elites))]
        remaining_parents_to_select = num_parents - len(selected_parents)
        parent_ids_so_far = {p.id for p in selected_parents}

        if settings.DEBUG:
            logger.debug(f"Selected {len(selected_parents)} elite parents globally: {[p.id for p in selected_parents]}")

        if remaining_parents_to_select <= 0:
            return selected_parents

        # Then fill remaining slots using roulette wheel selection from each island proportionally
        # Collect all non-elite candidates from all islands
        all_roulette_candidates_with_island = []
        for island_id, island in self.islands.items():
            if not island.programs:
                continue
            
            # Sort island programs to pick non-elites for roulette
            sorted_island_programs = sorted(
                island.programs,
                key=lambda p: (p.fitness_scores.get("correctness", 0.0), -p.fitness_scores.get("runtime_ms", float('inf')), -p.generation),
                reverse=True
            )
            for program in sorted_island_programs:
                if program.id not in parent_ids_so_far: # Check if not already selected as an elite
                    all_roulette_candidates_with_island.append(program)
        
        # Deduplicate candidates for roulette (if a program migrated and is not an elite)
        unique_roulette_candidates = []
        seen_roulette_ids = set()
        for cand in all_roulette_candidates_with_island:
            if cand.id not in parent_ids_so_far and cand.id not in seen_roulette_ids:
                 unique_roulette_candidates.append(cand)
                 seen_roulette_ids.add(cand.id)

        if not unique_roulette_candidates:
            logger.warning("No more unique candidates for roulette selection.")
            return selected_parents

        # Perform roulette wheel selection on the combined pool of unique candidates
        total_fitness_roulette = sum(p.fitness_scores.get("correctness", 0.0) + 0.0001 for p in unique_roulette_candidates) # Add small constant to allow selection of 0 fitness programs

        if total_fitness_roulette <= 0.0001 * len(unique_roulette_candidates): # All candidates have near zero fitness
            num_to_select_randomly = min(remaining_parents_to_select, len(unique_roulette_candidates))
            random_parents_from_roulette = random.sample(unique_roulette_candidates, num_to_select_randomly)
            selected_parents.extend(random_parents_from_roulette)
            logger.debug(f"Selected {len(random_parents_from_roulette)} random parents due to low/zero total fitness in roulette pool.")
        else:
            for _ in range(remaining_parents_to_select):
                if not unique_roulette_candidates: break
                pick = random.uniform(0, total_fitness_roulette)
                current_sum = 0
                chosen_parent = None
                for i, program in enumerate(unique_roulette_candidates):
                    current_sum += (program.fitness_scores.get("correctness", 0.0) + 0.0001)
                    if current_sum >= pick:
                        chosen_parent = program
                        unique_roulette_candidates.pop(i) # Remove chosen parent from further selection
                        total_fitness_roulette -= (chosen_parent.fitness_scores.get("correctness", 0.0) + 0.0001) # Adjust total fitness
                        break
                
                if chosen_parent:
                    selected_parents.append(chosen_parent)
                    parent_ids_so_far.add(chosen_parent.id) # Track for debugging or future use
                    logger.debug(f"Selected parent via global roulette: {chosen_parent.id} from island {chosen_parent.island_id} with correctness {chosen_parent.fitness_scores.get('correctness')}")
                elif unique_roulette_candidates: # Fallback if pick logic fails (should not happen with correct total_fitness adjustment)
                    fallback_parent = random.choice(unique_roulette_candidates)
                    selected_parents.append(fallback_parent)
                    unique_roulette_candidates.remove(fallback_parent)
                    parent_ids_so_far.add(fallback_parent.id)
                    logger.debug(f"Selected fallback parent via global roulette: {fallback_parent.id}")
        
        logger.info(f"Selected {len(selected_parents)} parents in total.")
        return selected_parents

    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int) -> List[Program]:
        """
        Select survivors for each island, combining current island members with their offspring.
        as island.programs is the source of truth for each island's current members.
        """
        if settings.DEBUG:
            logger.debug(f"Starting survivor selection. Offspring pop: {len(offspring_population)}, Target pop size: {population_size}")
        
        # Update island metrics
        for island in self.islands.values():
            island.update_metrics()

        # Check if it's time for migration
        if self.current_generation % self.migration_interval == 0:
            if settings.DEBUG:
                logger.debug(f"Generation {self.current_generation}: Performing migration")
            self._perform_migration()

        self.current_generation += 1
        if settings.DEBUG:
            logger.debug(f"Generation incremented to {self.current_generation}")

        # Select survivors within each island
        all_survivors = []
        programs_per_island = population_size // self.num_islands

        for island_id, island in self.islands.items():
            if settings.DEBUG:
                logger.debug(f"Processing Island {island_id} for survivor selection")
            
            # Get current island members
            current_island_members = island.programs
            
            # Filter offspring belonging to this island
            newly_generated_for_this_island = [
                p for p in offspring_population if p.island_id == island_id
            ]
            
            if settings.DEBUG:
                logger.debug(f"Island {island_id}: {len(current_island_members)} current members, "
                           f"{len(newly_generated_for_this_island)} new offspring")
            
            combined_population = current_island_members + newly_generated_for_this_island
            if not combined_population:
                island.programs = []  # Island becomes empty
                if settings.DEBUG:
                    logger.debug(f"Island {island_id} became empty")
                continue

            # Sort by correctness (higher is better), runtime (lower is better), and generation (lower/older is better)
            sorted_combined = sorted(
                combined_population,
                key=lambda p: (
                    p.fitness_scores.get("correctness", 0.0),  # Higher correctness preferred
                    -p.fitness_scores.get("runtime_ms", float('inf')),  # Lower runtime preferred
                    -p.generation  # Older generation preferred
                ),
                reverse=True
            )

            survivors = []
            seen_program_ids = set()
            for program in sorted_combined:
                if len(survivors) < programs_per_island:
                    if program.id not in seen_program_ids:
                        survivors.append(program)
                        seen_program_ids.add(program.id)
                        if settings.DEBUG:
                            logger.debug(f"Island {island_id} selected survivor: {program.id} "
                                       f"with correctness {program.fitness_scores.get('correctness')}")
                else:
                    break

            island.programs = survivors
            all_survivors.extend(survivors)
            if settings.DEBUG:
                logger.debug(f"Island {island_id} final survivor count: {len(survivors)}")

        return all_survivors

    def _perform_migration(self) -> None:
        """Perform migration between islands."""
        if settings.DEBUG:
            logger.debug("Starting migration process")
        
        # Identify underperforming islands
        island_performances = [(island_id, island.get_best_program().fitness_scores.get("correctness", 0.0) if island.get_best_program() else 0.0) 
                             for island_id, island in self.islands.items()]
        sorted_islands = sorted(island_performances, key=lambda x: x[1])
        
        # Select the worst performing half of islands
        num_islands_to_reseed = self.num_islands // 2
        underperforming_islands = [island_id for island_id, _ in sorted_islands[:num_islands_to_reseed]]
        surviving_islands = [island_id for island_id, _ in sorted_islands[num_islands_to_reseed:]]
        
        if settings.DEBUG:
            logger.debug(f"Identified {len(underperforming_islands)} underperforming islands: {underperforming_islands}")
            logger.debug(f"Identified {len(surviving_islands)} surviving islands: {surviving_islands}")
        
        # Get the best programs from surviving islands
        for underperforming_island_id in underperforming_islands:
            if not surviving_islands: # Should not happen if num_islands > 1
                logger.warning("No surviving islands to donate for migration.")
                break

            # Select a random surviving island
            donor_island_id = random.choice(surviving_islands)
            donor_island = self.islands[donor_island_id]
            recipient_island = self.islands[underperforming_island_id]
            
            # Get the best program from the donor island
            best_program_from_donor = donor_island.get_best_program()
            if best_program_from_donor:
                # Create a deep copy or a new Program instance to avoid shared object issues if necessary,
                # especially if the program object might be modified later independently by islands.
                # For now, assuming Program objects are relatively immutable post-evaluation or that sharing is acceptable.
                migrant_program = best_program_from_donor # Potentially clone this: copy.deepcopy(best_program_from_donor)
                migrant_program.island_id = underperforming_island_id # Assign to new island
                # Reset generation if it's a pure re-seed, or keep if it's a true 'migration' maintaining age.
                # For this less destructive migration, let's assume it keeps its generation from the donor.

                # Add to recipient island's program list, ensuring no duplicates by ID
                if not any(p.id == migrant_program.id for p in recipient_island.programs):
                    recipient_island.programs.append(migrant_program)
                    if settings.DEBUG:
                        logger.debug(f"Migrated program {migrant_program.id} (Correctness: {migrant_program.fitness_scores.get('correctness')}) "
                                   f"from island {donor_island_id} to island {underperforming_island_id}. "
                                   f"Recipient island size now: {len(recipient_island.programs)}")
                else:
                    if settings.DEBUG:
                        logger.debug(f"Program {migrant_program.id} from donor island {donor_island_id} already exists in recipient {underperforming_island_id}. Skipped migration of this specific program.")
            else:
                if settings.DEBUG:
                    logger.debug(f"Donor island {donor_island_id} had no best program to migrate.")

    async def execute(self, action: str, **kwargs) -> Any:
        # This method is part of the BaseAgent interface.
        # Specific actions like initialize_islands, select_parents, select_survivors
        # are called directly. If other generic async actions are needed for
        # SelectionControllerAgent in the future, they can be dispatched here.
        logger.warning(f"SelectionControllerAgent.execute called with action '{action}', but most actions are handled by specific methods.")
        if action == "initialize_islands_async_placeholder": # Example if an async version was needed
            # await self.async_initialize_islands(kwargs['initial_programs'])
            pass
        raise NotImplementedError(f"The generic execute method is not fully implemented for specific action '{action}' in SelectionControllerAgent. Use direct methods.")

                
if __name__ == '__main__':
    import uuid
    import random
    logging.basicConfig(level=logging.DEBUG)
    selector = SelectionControllerAgent()

    # Create test programs with proper attributes
    programs = [
        Program(
            id=str(uuid.uuid4()),
            code="c1",
            fitness_scores={"correctness": 0.9, "runtime_ms": 100},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c2",
            fitness_scores={"correctness": 1.0, "runtime_ms": 50},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c3",
            fitness_scores={"correctness": 0.7, "runtime_ms": 200},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c4",
            fitness_scores={"correctness": 1.0, "runtime_ms": 60},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c5",
            fitness_scores={"correctness": 0.5},
            status="evaluated",
            generation=0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="c6",
            status="unevaluated",
            generation=0
        ),
    ]

    # Initialize islands
    selector.initialize_islands(programs)
    print("\n--- Initial Island Distribution ---")
    for island_id, island in selector.islands.items():
        print(f"Island {island_id}: {len(island.programs)} programs")
        for p in island.programs:
            print(f"  Program {p.id}: Gen={p.generation}, Correctness={p.fitness_scores.get('correctness')}, Runtime={p.fitness_scores.get('runtime_ms')}")

    print("\n--- Testing Parent Selection ---")
    parents = selector.select_parents(programs, num_parents=3)
    for p in parents:
        print(f"Selected Parent: {p.id}, Island: {p.island_id}, Gen: {p.generation}, Correctness: {p.fitness_scores.get('correctness')}, Runtime: {p.fitness_scores.get('runtime_ms')}")

    print("\n--- Testing Survivor Selection ---")
    current_pop = programs[:2]
    offspring_pop = [
        Program(
            id=str(uuid.uuid4()),
            code="off1",
            fitness_scores={"correctness": 1.0, "runtime_ms": 40},
            status="evaluated",
            generation=1,
            island_id=0  # Simulate offspring from island 0
        ),
        Program(
            id=str(uuid.uuid4()),
            code="off2",
            fitness_scores={"correctness": 0.6, "runtime_ms": 10},
            status="evaluated",
            generation=1,
            island_id=1  # Simulate offspring from island 1
        ),
    ]
    
    # Simulate multiple generations to test migration
    for gen in range(3):
        print(f"\n--- Generation {gen} ---")
        survivors = selector.select_survivors(current_pop, offspring_pop, population_size=2)
        print(f"Survivors after generation {gen}:")
        for s in survivors:
            print(f"  Survivor: {s.id}, Island: {s.island_id}, Gen: {s.generation}, Correctness: {s.fitness_scores.get('correctness')}, Runtime: {s.fitness_scores.get('runtime_ms')}")
        
        # Update current population for next generation
        current_pop = survivors
        # Create new offspring with incremented generation
        # Note: gen + 2 is correct because:
        # - gen starts at 0
        # - selector.current_generation is incremented in select_survivors
        # - So offspring for next generation should be (current_generation + 1)
        offspring_pop = [
            Program(
                id=str(uuid.uuid4()),
                code=f"off{gen}_{i}",
                fitness_scores={"correctness": random.uniform(0.5, 1.0), "runtime_ms": random.randint(10, 200)},
                status="evaluated",
                generation=gen + 2,  # Correct generation for next generation
                island_id=i % selector.num_islands
            )
            for i in range(2)
        ] 