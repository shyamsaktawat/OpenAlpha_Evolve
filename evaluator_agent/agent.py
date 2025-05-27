                  
import asyncio
import json
import logging
import math
import os
import sys
import tempfile
import time
from typing import Optional, Dict, Any, Tuple, List

from config import settings
from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent

logger = logging.getLogger(__name__)

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        super().__init__()
        self.task_definition = task_definition
        self.evaluation_model_name = settings.EVALUATION_MODEL
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS
        logger.info(f"EvaluatorAgent initialized with model: {self.evaluation_model_name}, timeout: {self.evaluation_timeout_seconds}s")
        if self.task_definition:
            logger.info(f"EvaluatorAgent task_definition: {self.task_definition.id}")

    def _check_syntax(self, code: str) -> List[str]:
        errors = []
        try:
            compile(code+"\n", "tmp.py", 'exec')
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}")
        except Exception as e:
            errors.append(f"Unexpected error during syntax check: {str(e)}")
        return errors

    async def _execute_code_safely(
        self,
        code: str,
        task_for_examples: TaskDefinition,
        timeout_seconds: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        timeout = timeout_seconds if timeout_seconds is not None else self.evaluation_timeout_seconds
        results = {"test_outputs": [], "average_runtime_ms": 0.0}

        if not task_for_examples.input_output_examples:
            logger.warning("No input/output examples provided to _execute_code_safely.")
            return results, "No test cases to run."

        if not task_for_examples.function_name_to_evolve:
            logger.error(f"Task {task_for_examples.id} does not specify 'function_name_to_evolve'. Cannot execute code.")
            return None, "Task definition is missing 'function_name_to_evolve'."

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_script.py")

        def serialize_arg(arg):
            if isinstance(arg, (float, int)) and (arg == float('inf') or arg == float('-inf') or arg != arg):
                return f"float('{str(arg)}')"
            return json.dumps(arg)


        test_cases_str = json.dumps(task_for_examples.input_output_examples)
        test_cases_str = test_cases_str.replace('"Infinity"', 'float("inf")')
        test_cases_str = test_cases_str.replace('"-Infinity"', 'float("-inf")')
        test_cases_str = test_cases_str.replace('"NaN"', 'float("nan")')

        test_cases_str = test_cases_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')

        test_harness_code = f"""
import json
import time
import sys
import math  # Import math for inf/nan constants

# User's code (function to be tested)
{code}

# Test execution logic
results = []
total_execution_time = 0
num_tests = 0

# Special constants for test cases
Infinity = float('inf')
NaN = float('nan')

test_cases = {test_cases_str} 
function_to_test_name = "{task_for_examples.function_name_to_evolve}"

# Make sure the function_to_test is available in the global scope
if function_to_test_name not in globals():
    # Attempt to find it if it was defined inside a class (common for LLM output)
    # This is a simple heuristic and might need refinement.
    found_func = None
    for name, obj in list(globals().items()):
        if isinstance(obj, type):
            if hasattr(obj, function_to_test_name):
                method = getattr(obj, function_to_test_name)
                if callable(method):
                    globals()[function_to_test_name] = method
                    found_func = True
                    break
    if not found_func:
        print(json.dumps({{"error": f"Function '{{function_to_test_name}}' not found in the global scope or as a callable method of a defined class."}}))
        sys.exit(1)
        
function_to_test = globals()[function_to_test_name]

for i, test_case in enumerate(test_cases):
    input_args = test_case.get("input")
    
    start_time = time.perf_counter()
    try:
        if isinstance(input_args, list):
            actual_output = function_to_test(*input_args)
        elif isinstance(input_args, dict):
            actual_output = function_to_test(**input_args)
        elif input_args is None:
            actual_output = function_to_test()
        else:
            actual_output = function_to_test(input_args)
            
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        total_execution_time += execution_time_ms
        num_tests += 1
        results.append({{"test_case_id": i, "output": actual_output, "runtime_ms": execution_time_ms, "status": "success"}})
    except Exception as e:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        error_output = {{
            "test_case_id": i,
            "error": str(e), 
            "error_type": type(e).__name__,
            "runtime_ms": execution_time_ms,
            "status": "error"
        }}
        try:
            json.dumps(error_output)
        except TypeError:
            error_output["error"] = "Unserializable error object"
        results.append(error_output)

final_output = {{"test_outputs": results}}
if num_tests > 0:
    final_output["average_runtime_ms"] = total_execution_time / num_tests

def custom_json_serializer(obj):
    if isinstance(obj, float):
        if obj == float('inf'):
            return 'Infinity'
        elif obj == float('-inf'):
            return '-Infinity'
        elif obj != obj:
            return 'NaN'
    raise TypeError(f"Object of type {{type(obj).__name__}} is not JSON serializable")

print(json.dumps(final_output, default=custom_json_serializer))
"""
        with open(temp_file_path, "w") as f:
            f.write(test_harness_code)

        # Generate a unique container name to manage it during timeouts
        container_name = f"evaluator-{task_for_examples.id}-{time.time_ns()}"
        
        # Docker command construction
        cmd = [
            "docker", "run",
            "--rm",
            "--name", container_name,
            "-i",
            # Conditionally disable network
            # "-v", f"{os.path.abspath(temp_dir)}:/app/user_code", # Ensure absolute path for temp_dir # This line will be part of the dynamic extension below
            "-w", "/app/user_code",
            # settings.DOCKER_IMAGE_NAME, # This line will be part of the dynamic extension below
            # "python", "temp_script.py" # This line will be part of the dynamic extension below
        ]

        if settings.DOCKER_NETWORK_DISABLED:
            cmd.extend(["--network", "none"])
        
        # Add volume mount, image name, and script execution command
        cmd.extend([
            "-v", f"{os.path.abspath(temp_dir)}:/app/user_code",
            settings.DOCKER_IMAGE_NAME,
            "python", "temp_script.py"
        ])

        proc = None
        try:
            logger.debug(f"Executing code in Docker: {' '.join(cmd)}")
            start_time = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.monotonic() - start_time
            logger.debug(f"Docker execution finished in {duration:.2f}s. Exit code: {proc.returncode}")

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            if proc.returncode != 0:
                # If stdout is empty and stderr has content, it's likely a Docker/script init error
                if not stdout_str and stderr_str:
                    error_message = f"Execution failed with exit code {proc.returncode}. Docker error: '{stderr_str}'"
                    logger.warning(error_message)
                    return None, error_message
                # If stdout has content, it might be a script error with traceback in stderr, but JSON in stdout.
                # Log a warning and proceed to parse stdout. If parsing fails, that error will be returned.
                logger.warning(f"Execution completed with non-zero exit code {proc.returncode}. Stdout: '{stdout_str}', Stderr: '{stderr_str}'. Attempting to parse stdout.")

            if not stdout_str and proc.returncode == 0: # Script exited cleanly but no output
                 logger.warning(f"Execution produced no stdout, but exited cleanly. Stderr: '{stderr_str}'")
                 return None, f"No output from script. Stderr: '{stderr_str}'"
            
            if not stdout_str and proc.returncode != 0: # No stdout and non-zero exit, means previous error message should be used
                 return None, f"Execution failed with exit code {proc.returncode}. No stdout. Stderr: '{stderr_str}'"


            try:
                def json_loads_with_infinity(s):
                    s = s.replace('"Infinity"', 'float("inf")')
                    s = s.replace('"-Infinity"', 'float("-inf")')
                    s = s.replace('"NaN"', 'float("nan")')
                    return json.loads(s)

                parsed_output = json_loads_with_infinity(stdout_str)
                logger.debug(f"Parsed execution output: {parsed_output}")
                return parsed_output, None
            except json.JSONDecodeError as e:
                error_message = f"Failed to decode JSON output: {e}. Raw output: '{stdout_str}'"
                logger.error(error_message)
                return None, error_message
            except Exception as e:
                error_message = f"Error processing script output: {e}. Raw output: '{stdout_str}'"
                logger.error(error_message)
                return None, error_message

        except asyncio.TimeoutError:
            logger.warning(f"Execution for container '{container_name}' initiating timeout handling.")
            if proc and proc.returncode is None: # Check if process is still running
                logger.info(f"Attempting to stop Docker container: {container_name}")
                stop_cmd = ["docker", "stop", container_name]
                try:
                    stop_proc = await asyncio.create_subprocess_exec(*stop_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    _, stop_stderr_bytes = await asyncio.wait_for(stop_proc.communicate(), timeout=10) # 10s for docker stop
                    if stop_proc.returncode != 0:
                        logger.error(f"Failed to stop container {container_name}. Exit: {stop_proc.returncode}. Stderr: {stop_stderr_bytes.decode(errors='replace')}")
                        kill_cmd = ["docker", "kill", container_name]
                        kill_proc = await asyncio.create_subprocess_exec(*kill_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                        kill_stdout_bytes, kill_stderr_bytes = await asyncio.wait_for(kill_proc.communicate(), timeout=5) # 5s for docker kill
                        if kill_proc.returncode == 0:
                             logger.info(f"Successfully killed container {container_name} after stop failed.")
                        else:
                             logger.error(f"Failed to kill container {container_name}. Exit: {kill_proc.returncode}. Stderr: {kill_stderr_bytes.decode(errors='replace')}")
                    else:
                        logger.info(f"Successfully stopped container {container_name}.")
                except asyncio.TimeoutError:
                    logger.error(f"Timeout trying to stop/kill container {container_name}. It might be orphaned.")
                except Exception as e_stop:
                    logger.error(f"Error stopping/killing container {container_name}: {e_stop}")
            
            if proc: # Original docker run process
                try:
                    if proc.returncode is None: proc.kill()
                    await proc.wait() 
                except ProcessLookupError: pass
                except Exception as e_kill: logger.error(f"Error trying to kill original subprocess after docker stop/kill: {e_kill}")
            
            logger.warning(f"Code execution in Docker container '{container_name}' timed out after {timeout} seconds.")
            return None, f"Execution timed out after {timeout} seconds (container {container_name})."
        except Exception as e:
            logger.error(f"An unexpected error occurred during code execution: {e}", exc_info=True)
            return None, f"Unexpected execution error: {str(e)}"
        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(temp_dir):
                    try:
                        # Attempt to remove the directory multiple times with a small delay
                        # This is a workaround for potential lingering locks from Docker
                        for _ in range(3):
                            try:
                                if os.path.exists(temp_file_path): os.remove(temp_file_path)
                                os.rmdir(temp_dir)
                                break # Succeeded
                            except OSError:
                                await asyncio.sleep(0.1) # Wait a bit and retry
                        else:
                            logger.error(f"Failed to remove temp_dir {temp_dir} after multiple retries.")
                    except Exception as e_rmdir: # Catch any other exception during rmdir attempts
                         logger.error(f"Error removing temp_dir {temp_dir}: {e_rmdir}.")
            except Exception as e_cleanup: # General cleanup exception
                logger.error(f"Error during cleanup of temp files: {e_cleanup}")

    def _assess_correctness(self, execution_results: Dict[str, Any], expected_outputs: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        passed_tests = 0
        total_tests = len(expected_outputs)

        if not execution_results or "test_outputs" not in execution_results:
            logger.warning("Execution results are missing 'test_outputs' field.")
            return 0.0, 0, total_tests

        actual_test_outputs = execution_results["test_outputs"]

        if len(actual_test_outputs) != total_tests:
            logger.warning(f"Mismatch in number of test outputs ({len(actual_test_outputs)}) and expected outputs ({total_tests}). Some tests might have crashed before producing output.")

        for i, expected in enumerate(expected_outputs):
            actual_output_detail = next((res for res in actual_test_outputs if res.get("test_case_id") == i), None)

            if actual_output_detail and actual_output_detail.get("status") == "success":
                actual = actual_output_detail.get("output")

                logger.debug(f"Test case {i}: Actual output: {actual}")

                # Check if we have a validation function
                if "validation_func" in expected:
                    logger.debug(f"Test case {i}: Using validation function.")
                    try:
                        # Create a namespace for the validation function
                        namespace = {}
                        # Execute the validation function definition
                        exec(expected["validation_func"], namespace)
                        # Get the validate function
                        validate_func = namespace.get("validate")
                        if validate_func and callable(validate_func):
                            # Revert: Only pass the actual output to the validation function
                            if validate_func(actual):
                                passed_tests += 1
                                logger.debug(f"Test case {i}: Validation function returned True.")
                            else:
                                logger.debug(f"Test case {i}: Validation function returned False.")
                        else:
                            logger.warning(f"Validation function not found or not callable in test case {i}")
                    except Exception as e:
                        logger.error(f"Error executing validation function for test case {i}: {str(e)}", exc_info=True) # Log exception details
                # Check against expected output if provided
                elif "output" in expected:
                    expected_val = expected["output"]
                    logger.debug(f"Test case {i}: Comparing with expected output: {expected_val}")
                    if self._compare_outputs(actual, expected_val):
                        passed_tests += 1
                        logger.debug(f"Test case {i}: Comparison returned True.")
                    else:
                        logger.debug(f"Test case {i}: Comparison returned False.")
                else:
                    logger.warning(f"Test case {i} has neither validation function nor expected output")
            elif actual_output_detail:
                logger.debug(f"Test case {i} had error: {actual_output_detail.get('error')}")
            else:
                logger.debug(f"Test case {i}: No output found in results.")

        logger.debug(f"Finished assessing correctness. Passed tests: {passed_tests}/{total_tests}")
        if total_tests == 0:
            return 1.0, 0, 0

        correctness = passed_tests / total_tests
        return correctness, passed_tests, total_tests

    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        logger.info(f"Evaluating program: {program.id} for task: {task.id}")
        program.status = "evaluating"
        program.errors = []
        program.fitness_scores = {"correctness": 0.0, "runtime_ms": float('inf'), "passed_tests": 0.0, "total_tests": 0.0}

        syntax_errors = self._check_syntax(program.code)
        if syntax_errors:
            program.errors.extend(syntax_errors)
            program.fitness_scores["correctness"] = 0.0
            program.status = "failed_evaluation"
            logger.warning(f"Syntax errors found in program {program.id}: {syntax_errors}")
            return program

        logger.debug(f"Syntax check passed for program {program.id}.")

        overall_passed_tests = 0
        overall_total_tests = 0
        last_successful_level_avg_runtime = float('inf')
        highest_level_passed = -1

        # Determine test sets to run
        test_groups_to_run = []
        if task.tests: # New structure with levels
            # Sort test groups by level, defaulting level to 0 if not specified
            sorted_test_groups = sorted(task.tests, key=lambda g: g.get('level', 0))
            for group in sorted_test_groups:
                test_groups_to_run.append({
                    "name": group.get('name', f"level_{group.get('level', 0)}"),
                    "level": group.get('level', 0),
                    "test_cases": group.get('test_cases', [])
                })
        elif task.input_output_examples: # Fallback for old structure
            logger.warning(f"Task {task.id} uses legacy 'input_output_examples'. Consider migrating to 'tests' with levels.")
            test_groups_to_run.append({
                "name": "default_level",
                "level": 0,
                "test_cases": task.input_output_examples
            })
        
        if not test_groups_to_run:
            logger.info(f"No tests or input/output examples provided for task {task.id}. Skipping execution.")
            program.fitness_scores["correctness"] = 0.5 # Default score if no tests
            program.fitness_scores["runtime_ms"] = 0.0
            program.status = "evaluated" # No tests to fail
            return program

        for group_idx, test_group in enumerate(test_groups_to_run):
            level_name = test_group['name']
            current_level = test_group['level']
            current_level_test_cases = test_group['test_cases']

            if not current_level_test_cases:
                logger.info(f"Test group '{level_name}' (Level {current_level}) has no test cases. Skipping.")
                continue

            logger.info(f"Executing program {program.id} against test group '{level_name}' (Level {current_level}) with {len(current_level_test_cases)} test cases.")
            
            # Create a temporary TaskDefinition subset for _execute_code_safely and _assess_correctness
            temp_task_def_for_level = TaskDefinition(
                id=f"{task.id}_level_{current_level}",
                description=task.description, # Not directly used by execution, but good to have
                function_name_to_evolve=task.function_name_to_evolve,
                input_output_examples=current_level_test_cases, # This is the key part
                allowed_imports=task.allowed_imports # Also important for execution context
            )

            execution_results, execution_error = await self._execute_code_safely(program.code, task_for_examples=temp_task_def_for_level)
            
            if execution_error:
                logger.warning(f"Execution error for program {program.id} at level {current_level} ('{level_name}'): {execution_error}")
                program.errors.append(f"Execution Error at Level {current_level} ('{level_name}'): {execution_error}")
                program.status = "failed_evaluation"
                break # Stop evaluation cascade
            
            if execution_results is None:
                logger.warning(f"No execution results for program {program.id} at level {current_level} ('{level_name}').")
                program.errors.append(f"Execution Error: No results at Level {current_level} ('{level_name}').")
                program.status = "failed_evaluation"
                break # Stop evaluation cascade

            level_correctness, level_passed_tests, level_total_tests = self._assess_correctness(execution_results, current_level_test_cases)
            
            overall_passed_tests += level_passed_tests
            overall_total_tests += level_total_tests

            current_level_avg_runtime = execution_results.get("average_runtime_ms", float('inf'))
            if not isinstance(current_level_avg_runtime, (float, int)):
                current_level_avg_runtime = float('inf')

            logger.info(f"Program {program.id} Level {current_level} ('{level_name}') Correctness: {level_correctness:.2f} ({level_passed_tests}/{level_total_tests}), Avg Runtime: {current_level_avg_runtime}ms")

            if level_correctness < 1.0:
                error_msg = f"Failed {level_total_tests - level_passed_tests} of {level_total_tests} tests at Level {current_level} ('{level_name}')."
                program.errors.append(error_msg)
                program.status = "failed_evaluation"
                # Update overall fitness scores with results up to this failing level
                program.fitness_scores["correctness"] = overall_passed_tests / overall_total_tests if overall_total_tests > 0 else 0.0
                program.fitness_scores["passed_tests"] = float(overall_passed_tests)
                program.fitness_scores["total_tests"] = float(overall_total_tests)
                # Runtime is tricky here. Could be avg of last successful, or sum. Let's keep last successful level's.
                program.fitness_scores["runtime_ms"] = last_successful_level_avg_runtime 
                break # Stop evaluation cascade
            else:
                highest_level_passed = current_level
                last_successful_level_avg_runtime = current_level_avg_runtime
                # If this is the last group and all passed, program status will be evaluated.
                if group_idx == len(test_groups_to_run) - 1:
                    program.status = "evaluated" 
        
        # Final fitness score calculation after loop (if not broken early)
        if overall_total_tests > 0:
            program.fitness_scores["correctness"] = overall_passed_tests / overall_total_tests
        elif not program.errors: # No tests run, but no syntax errors either
            program.fitness_scores["correctness"] = 0.5 # Default for no tests executed successfully
        # else correctness remains 0.0 from initialization if errors occurred before any tests
        
        program.fitness_scores["passed_tests"] = float(overall_passed_tests)
        program.fitness_scores["total_tests"] = float(overall_total_tests)
        program.fitness_scores["runtime_ms"] = last_successful_level_avg_runtime if highest_level_passed != -1 else float('inf')
        program.fitness_scores["highest_level_passed"] = float(highest_level_passed)

        # Consolidate status based on errors and correctness
        if program.errors:
            program.status = "failed_evaluation"
        elif program.fitness_scores.get("correctness", 0.0) == 1.0 and overall_total_tests > 0:
            program.status = "evaluated"
        elif overall_total_tests == 0 and not program.errors: # No tests were run (e.g. all groups empty), no syntax errors
             program.status = "evaluated" # Considered evaluated as there was nothing to fail
             program.fitness_scores["correctness"] = 0.5 # Re-affirm default
             program.fitness_scores["runtime_ms"] = 0.0
        elif not program.errors : # Some tests run, but not 100% correct, and no specific execution errors
            program.status = "failed_evaluation"
            if f"Achieved {program.fitness_scores['correctness']*100:.0f}% correctness but not all tests passed." not in program.errors:
                 program.errors.append(f"Achieved {program.fitness_scores['correctness']*100:.0f}% correctness but not all tests passed.")

        logger.info(f"Overall evaluation complete for program {program.id}. Status: {program.status}, Fitness: {program.fitness_scores}")
        return program

    async def execute(self, program: Program, task: TaskDefinition) -> Program:
        return await self.evaluate_program(program, task)

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        logger.debug(f"Comparing outputs. Actual: {type(actual)}{actual}, Expected: {type(expected)}{expected}")
        
        if isinstance(actual, float) and isinstance(expected, float):
            TOLERANCE = 1e-9 # This could also be made configurable via settings.py later.
            is_close = math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
            if not is_close:
                logger.debug(f"Float comparison: {actual} vs {expected} is NOT close (tolerance: {TOLERANCE}).")
            return is_close
        
        # Fallback to direct equality for other types
        are_equal = actual == expected

        return are_equal

                                                 
                                                              
                                                                                              
                                                         
                                                        
                                                                    