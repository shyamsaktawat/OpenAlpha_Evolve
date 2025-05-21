import logging
import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional

from core.interfaces import DatabaseAgentInterface, Program, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS programs (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    generation INTEGER,
    parent_id TEXT,
    island_id INTEGER,
    status TEXT,
    errors TEXT, 
    fitness_scores TEXT,
    behavioral_descriptors TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_id TEXT 
);
"""

class SQLiteDatabaseAgent(DatabaseAgentInterface, BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_path = settings.DATABASE_PATH
        self._initialize_db()
        logger.info(f"SQLiteDatabaseAgent initialized with db path: {self.db_path}")

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(DB_SCHEMA)
                conn.commit()
            logger.info("Database initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)
            raise

    def save_program(self, program: Program) -> None:
        logger.debug(f"Saving program {program.id} to SQLite.")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                errors_json = json.dumps(program.errors)
                fitness_scores_json = json.dumps(program.fitness_scores)
                behavioral_descriptors_json = json.dumps(program.behavioral_descriptors)
                
                # Handle potential task_id, defaulting if not present
                task_id = getattr(program, 'task_id', program.__dict__.get('task_id', 'N/A'))


                cursor.execute("""
                    INSERT OR REPLACE INTO programs 
                    (id, code, generation, parent_id, island_id, status, errors, fitness_scores, behavioral_descriptors, task_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    program.id, program.code, program.generation, program.parent_id,
                    program.island_id, program.status, errors_json, fitness_scores_json,
                    behavioral_descriptors_json, task_id
                ))
                conn.commit()
            logger.info(f"Program {program.id} saved successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error saving program {program.id}: {e}", exc_info=True)
        except json.JSONDecodeError as e:
            logger.error(f"Error serializing program data for {program.id}: {e}", exc_info=True)


    def get_program(self, program_id: str) -> Optional[Program]:
        logger.debug(f"Attempting to retrieve program {program_id} from SQLite.")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, code, generation, parent_id, island_id, status, errors, fitness_scores, behavioral_descriptors, task_id FROM programs WHERE id = ?", (program_id,))
                row = cursor.fetchone()

                if row:
                    program_data = {
                        "id": row[0],
                        "code": row[1],
                        "generation": row[2],
                        "parent_id": row[3],
                        "island_id": row[4],
                        "status": row[5],
                        "errors": json.loads(row[6]) if row[6] else [],
                        "fitness_scores": json.loads(row[7]) if row[7] else {},
                        "behavioral_descriptors": json.loads(row[8]) if row[8] else {},
                        # task_id is not part of Program dataclass by default
                    }
                    # If task_id needs to be on the object, it can be added here,
                    # or the Program class can be modified. For now, just log it.
                    retrieved_task_id = row[9] 
                    logger.debug(f"Retrieved program {program_id} with task_id: {retrieved_task_id}")
                    
                    program = Program(**program_data)
                    # Manually set task_id if needed, e.g., program.task_id = retrieved_task_id
                    # For now, we assume Program doesn't strictly define it but it's in DB
                    
                    logger.info(f"Program {program_id} retrieved successfully.")
                    return program
                else:
                    logger.info(f"Program {program_id} not found.")
                    return None
        except sqlite3.Error as e:
            logger.error(f"Error retrieving program {program_id}: {e}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error deserializing data for program {program_id}: {e}", exc_info=True)
            return None

    def get_all_programs(self) -> List[Program]:
        logger.debug("Attempting to retrieve all programs from SQLite.")
        programs = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, code, generation, parent_id, island_id, status, errors, fitness_scores, behavioral_descriptors, task_id FROM programs")
                rows = cursor.fetchall()

                for row in rows:
                    program_data = {
                        "id": row[0],
                        "code": row[1],
                        "generation": row[2],
                        "parent_id": row[3],
                        "island_id": row[4],
                        "status": row[5],
                        "errors": json.loads(row[6]) if row[6] else [],
                        "fitness_scores": json.loads(row[7]) if row[7] else {},
                        "behavioral_descriptors": json.loads(row[8]) if row[8] else {},
                    }
                    programs.append(Program(**program_data))
            logger.info(f"Retrieved {len(programs)} programs successfully.")
            return programs
        except sqlite3.Error as e:
            logger.error(f"Error retrieving all programs: {e}", exc_info=True)
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error deserializing data during get_all_programs: {e}", exc_info=True)
            return [] # Or handle more gracefully

    def get_best_programs(self, task_id: str, limit: int = 5, objective: str = "correctness", sort_order: str = "desc") -> List[Program]:
        logger.debug(f"Retrieving best programs for task {task_id} by {objective} ({sort_order}), limit {limit}.")
        programs = []
        # Basic validation for sort_order to prevent SQL injection if it were less controlled
        order = "DESC" if sort_order.lower() == "desc" else "ASC"
        
        # This is a simplified way to sort by a JSON field.
        # For robust sorting, it's better if fitness scores are in separate columns or use JSON functions if SQLite version supports them.
        # Here, we fetch all and sort in Python if objective is complex, or do a simple sort if objective is a direct column.
        # For 'correctness', assuming it's a key in fitness_scores JSON.
        # SQLite's json_extract is good here if available and known path.
        # Example: ORDER BY json_extract(fitness_scores, '$.correctness') DESC
        
        query = f"""
            SELECT id, code, generation, parent_id, island_id, status, errors, fitness_scores, behavioral_descriptors, task_id 
            FROM programs
            WHERE task_id = ?
            ORDER BY created_at DESC -- Fallback sort, refine with json_extract if possible
            LIMIT ?
        """
        # The ORDER BY clause above is a placeholder. True sorting by a JSON key is more complex.
        # We'll sort in Python after fetching for simplicity for now for 'correctness'.

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Fetching more and sorting in Python if the objective is complex.
                # If objective is simple like 'generation', SQL sort is fine.
                if objective == "correctness": # Requires JSON parsing
                     cursor.execute("SELECT id, code, generation, parent_id, island_id, status, errors, fitness_scores, behavioral_descriptors, task_id FROM programs WHERE task_id = ?", (task_id,))
                else: # Assuming objective is a direct column like 'generation' or 'created_at'
                    # Ensure objective is a safe column name
                    safe_objective_column = "created_at" # default
                    if objective in ["generation", "status", "id"]: # Whitelist direct sortable columns
                        safe_objective_column = objective
                    
                    sort_query = f"""
                        SELECT id, code, generation, parent_id, island_id, status, errors, fitness_scores, behavioral_descriptors, task_id 
                        FROM programs
                        WHERE task_id = ?
                        ORDER BY {safe_objective_column} {order}
                        LIMIT ?
                    """
                    cursor.execute(sort_query, (task_id, limit))

                rows = cursor.fetchall()
                
                for row in rows:
                    try:
                        program_data = {
                            "id": row[0],
                            "code": row[1],
                            "generation": row[2],
                            "parent_id": row[3],
                            "island_id": row[4],
                            "status": row[5],
                            "errors": json.loads(row[6]) if row[6] else [],
                            "fitness_scores": json.loads(row[7]) if row[7] else {},
                            "behavioral_descriptors": json.loads(row[8]) if row[8] else {},
                        }
                        programs.append(Program(**program_data))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error deserializing program data for {row[0]} in get_best_programs: {e}")
                        continue # Skip this program

            # If sorting by a JSON field like 'correctness'
            if objective == "correctness":
                programs.sort(key=lambda p: p.fitness_scores.get(objective, 0 if order == "ASC" else float('-inf')), reverse=(order == "DESC"))
                programs = programs[:limit]

            logger.info(f"Retrieved {len(programs)} best programs for task {task_id}.")
            return programs

        except sqlite3.Error as e:
            logger.error(f"Error retrieving best programs for task {task_id}: {e}", exc_info=True)
            return []


    def count_programs(self) -> int:
        logger.debug("Counting all programs in SQLite.")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM programs")
                count = cursor.fetchone()[0]
                logger.info(f"Total programs count: {count}.")
                return count
        except sqlite3.Error as e:
            logger.error(f"Error counting programs: {e}", exc_info=True)
            return 0

    def clear_database(self) -> None:
        logger.info("Clearing all programs from the database.")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM programs")
                conn.commit()
            logger.info("Database cleared successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error clearing database: {e}", exc_info=True)

    # --- MAP-Elites specific methods (placeholders) ---
    def get_elites_by_task(self, task_id: str) -> List[Program]:
        logger.warning(f"MAP-Elites 'get_elites_by_task' not fully implemented in SQLiteDatabaseAgent for task {task_id}. Returning empty list.")
        return []

    def offer_to_archive(self, program: Program, scalar_fitness: float, task_id: str) -> bool:
        logger.warning(f"MAP-Elites 'offer_to_archive' not fully implemented in SQLiteDatabaseAgent for program {program.id}, task {task_id}. Returning False.")
        return False
        
    def execute(self, *args, **kwargs) -> Any:
        logger.error("The 'execute' method is not implemented for SQLiteDatabaseAgent.")
        raise NotImplementedError("This agent does not support generic execute commands.")

# Example Usage (Optional, for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mock settings for standalone testing
    class MockSettings:
        DATABASE_PATH = "test_sqlite_agent.db"
        # Add other settings if needed by the agent, though this one primarily uses DATABASE_PATH

    settings = MockSettings() # Override imported settings for this test block

    # Clean up previous test database if it exists
    import os
    if os.path.exists(settings.DATABASE_PATH):
        os.remove(settings.DATABASE_PATH)

    db_agent = SQLiteDatabaseAgent()

    # Test save_program
    test_program_id = str(uuid.uuid4())
    program1 = Program(
        id=test_program_id,
        code="def hello():\n  print('Hello, world!')",
        generation=1,
        parent_id=None,
        island_id=0,
        status="evaluated",
        errors=[],
        fitness_scores={"correctness": 0.95, "runtime_ms": 120},
        behavioral_descriptors={"output_length": 13, "has_loop": 0}
    )
    # Add task_id for saving, as it's in the schema
    program1.task_id = "test_task_001" 
    db_agent.save_program(program1)

    # Test get_program
    retrieved_program = db_agent.get_program(test_program_id)
    if retrieved_program:
        logger.info(f"Retrieved program: {retrieved_program.id}, Fitness: {retrieved_program.fitness_scores}")
        assert retrieved_program.id == test_program_id
        assert retrieved_program.fitness_scores.get("correctness") == 0.95
    else:
        logger.error("Failed to retrieve test program.")

    # Test get_all_programs
    all_programs = db_agent.get_all_programs()
    logger.info(f"Total programs found: {len(all_programs)}")
    assert len(all_programs) == 1

    # Test get_best_programs
    program2_id = str(uuid.uuid4())
    program2 = Program(
        id=program2_id,
        code="def another():\n  pass",
        generation=2,
        fitness_scores={"correctness": 0.99, "runtime_ms": 50},
        task_id="test_task_001" # Must be set for get_best_programs
    )
    db_agent.save_program(program2)
    
    program3_id = str(uuid.uuid4())
    program3 = Program(
        id=program3_id,
        code="def yet_another():\n  pass",
        generation=1,
        fitness_scores={"correctness": 0.90, "runtime_ms": 200},
        task_id="test_task_001"
    )
    db_agent.save_program(program3)

    best_programs = db_agent.get_best_programs(task_id="test_task_001", limit=2, objective="correctness", sort_order="desc")
    logger.info(f"Best programs for task test_task_001: {[p.id for p in best_programs]}")
    assert len(best_programs) == 2
    if best_programs:
      assert best_programs[0].id == program2_id # program2 has correctness 0.99

    # Test count_programs
    count = db_agent.count_programs()
    logger.info(f"Current program count: {count}")
    assert count == 3
    
    # Test MAP-Elites placeholders
    db_agent.get_elites_by_task("test_task_001")
    db_agent.offer_to_archive(program1, 0.95, "test_task_001")

    # Test clear_database
    # db_agent.clear_database()
    # count_after_clear = db_agent.count_programs()
    # logger.info(f"Program count after clear: {count_after_clear}")
    # assert count_after_clear == 0

    logger.info("SQLiteDatabaseAgent basic tests completed.")
    # Clean up test database
    # if os.path.exists(settings.DATABASE_PATH):
    #     os.remove(settings.DATABASE_PATH)
    #     logger.info(f"Test database {settings.DATABASE_PATH} removed.")

    # Test error handling for save_program (e.g., non-serializable data)
    # This is harder to test without modifying Program or passing bad data directly
    # For now, assume JSON serialization errors are caught by the try-except block.

    # Test what happens if DB path is invalid (e.g. permissions) - _initialize_db should raise
    # try:
    #     MockSettings.DATABASE_PATH = "/root/no_permission.db" # Invalid path
    #     settings_invalid = MockSettings()
    #     invalid_db_agent = SQLiteDatabaseAgent() # This should fail
    # except Exception as e:
    #     logger.info(f"Correctly caught error for invalid DB path: {e}")

    # Reset path for other tests if any
    # settings.DATABASE_PATH = "test_sqlite_agent.db"
