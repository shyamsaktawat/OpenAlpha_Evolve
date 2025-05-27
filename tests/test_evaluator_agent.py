import asyncio
import json
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, call

# Ensure the test runner can find the modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluator_agent.agent import EvaluatorAgent
from core.interfaces import Program, TaskDefinition
from config import settings

# Helper to create a mock subprocess
def create_mock_subprocess(stdout_data, stderr_data, return_code, communicate_raises=None):
    proc = MagicMock(spec=asyncio.subprocess.Process)
    proc.returncode = return_code
    
    if communicate_raises:
        proc.communicate = AsyncMock(side_effect=communicate_raises)
    else:
        proc.communicate = AsyncMock(return_value=(stdout_data.encode(), stderr_data.encode()))
    
    # Mock wait, kill, etc. if needed for more complex scenarios, e.g., timeout
    proc.wait = AsyncMock()
    proc.kill = MagicMock() # For non-async kill if called directly
    return proc

class TestEvaluatorAgentDockerExecution(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.agent = EvaluatorAgent()
        self.program = Program(id="test_prog", code="def solve():\n  return 42", parent_id=None, fitness_scores={})
        self.task_definition = TaskDefinition(
            id="test_task",
            description="Test task",
            function_name_to_evolve="solve",
            input_output_examples=[
                {"input": [], "output": 42}
            ]
        )
        
        # Mock settings for Docker - can be overridden per test
        self.mock_settings_patcher = patch('evaluator_agent.agent.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.DOCKER_IMAGE_NAME = "test-eval-image:latest"
        self.mock_settings.DOCKER_NETWORK_DISABLED = True
        self.mock_settings.EVALUATION_TIMEOUT_SECONDS = 5 # Short timeout for tests

    async def asyncTearDown(self):
        self.mock_settings_patcher.stop()
        # Clean up temp files created by the agent if any (though mocks should prevent most)
        # This is tricky because tempfile.mkdtemp() is used.
        # For robust testing, tempfile.mkdtemp could also be patched.

    @patch('asyncio.create_subprocess_exec', new_callable=AsyncMock)
    async def test_execute_code_safely_success(self, mock_create_subprocess_exec):
        # --- Test Case: Successful execution ---
        # Expected output from the script (JSON string)
        expected_script_output = {
            "test_outputs": [{"test_case_id": 0, "output": 42, "runtime_ms": 10.0, "status": "success"}],
            "average_runtime_ms": 10.0
        }
        mock_proc_docker_run = create_mock_subprocess(json.dumps(expected_script_output), "", 0)
        mock_create_subprocess_exec.return_value = mock_proc_docker_run

        results, error = await self.agent._execute_code_safely(self.program.code, self.task_definition)

        self.assertIsNotNone(results)
        self.assertIsNone(error)
        self.assertEqual(results["average_runtime_ms"], 10.0)
        self.assertEqual(results["test_outputs"][0]["output"], 42)
        
        # Verify docker command
        args, _ = mock_create_subprocess_exec.call_args
        self.assertEqual(args[0], "docker")
        self.assertEqual(args[1], "run")
        self.assertIn(self.mock_settings.DOCKER_IMAGE_NAME, args)
        if self.mock_settings.DOCKER_NETWORK_DISABLED:
            self.assertIn("--network", args)
            self.assertIn("none", args)
        
        # Check for volume mount structure (path will be dynamic due to tempfile)
        # Example: -v /tmp/somerandomdir:/app/user_code
        volume_arg_index = -1
        for i, arg in enumerate(args):
            if arg == "-v" and ":/app/user_code" in args[i+1]:
                volume_arg_index = i
                break
        self.assertNotEqual(volume_arg_index, -1, "Volume mount for temp script not found in Docker command.")
        # Further check on temp_script.py in mounted dir could be done if tempfile is patched.

    @patch('asyncio.create_subprocess_exec', new_callable=AsyncMock)
    async def test_execute_code_safely_script_error(self, mock_create_subprocess_exec):
        # --- Test Case: Script error ---
        script_stderr = "Traceback (most recent call last):\n  File \"temp_script.py\", line X, in <module>\n    raise ValueError(\"Test script error\")\nValueError: Test script error"
        # Script fails, so stdout might be empty or contain partial/error JSON from the harness.
        # Docker itself succeeds (return_code 0), but script inside fails (indicated by its output and possibly non-zero exit code from python interpreter inside docker).
        # For this test, assume the script harness catches the error and prints an error JSON, and python exits 0.
        # Or, if the python interpreter itself exits non-zero, docker run might also reflect that.
        # Current agent code: if proc.returncode != 0 AND not stdout_str AND stderr_str -> "Docker error"
        # if proc.returncode !=0 AND stdout_str -> "Execution completed with non-zero exit code... Attempting to parse stdout."
        # if not stdout_str AND proc.returncode == 0 -> "No output from script."
        
        # Scenario: Python script exits with 1, prints error to stderr, NO valid JSON to stdout
        mock_proc_docker_run = create_mock_subprocess("", script_stderr, 1) # stdout, stderr, returncode
        mock_create_subprocess_exec.return_value = mock_proc_docker_run

        results, error = await self.agent._execute_code_safely(self.program.code, self.task_definition)

        self.assertIsNone(results)
        self.assertIsNotNone(error)
        self.assertIn("Execution failed with exit code 1", error)
        self.assertIn("Docker error", error) # Because stdout is empty
        self.assertIn("Test script error", error)


    @patch('asyncio.create_subprocess_exec', new_callable=AsyncMock)
    async def test_execute_code_safely_docker_error(self, mock_create_subprocess_exec):
        # --- Test Case: Docker error ---
        # e.g. Docker image not found, Docker daemon error
        docker_stderr = "Error: No such image: non_existent_image:latest"
        # Docker command itself fails, e.g., exit code 125
        mock_proc_docker_run = create_mock_subprocess("", docker_stderr, 125)
        mock_create_subprocess_exec.return_value = mock_proc_docker_run
        
        results, error = await self.agent._execute_code_safely(self.program.code, self.task_definition)

        self.assertIsNone(results)
        self.assertIsNotNone(error)
        self.assertIn("Execution failed with exit code 125", error)
        self.assertIn("Docker error", error) # Because stdout is empty
        self.assertIn(docker_stderr, error)

    @patch('asyncio.create_subprocess_exec', new_callable=AsyncMock)
    async def test_execute_code_safely_timeout(self, mock_create_subprocess_exec):
        # --- Test Case: Timeout ---
        # Simulate timeout on the initial 'docker run' command
        # The process is still running when timeout occurs, so returncode should be None
        mock_proc_docker_run = create_mock_subprocess("", "", None, communicate_raises=asyncio.TimeoutError("Simulated timeout"))
        
        # Mocks for subsequent 'docker stop' and 'docker kill' attempts
        mock_proc_docker_stop = create_mock_subprocess("container_id_stopped", "", 0) # stdout, stderr, returncode
        mock_proc_docker_kill = create_mock_subprocess("container_id_killed", "", 0) # Not always called

        # Configure side_effect to return different mocks for different calls
        # 1st call: docker run (times out)
        # 2nd call: docker stop
        # 3rd call (optional, if stop fails): docker kill
        mock_create_subprocess_exec.side_effect = [
            mock_proc_docker_run, 
            mock_proc_docker_stop,
            mock_proc_docker_kill 
        ]

        results, error = await self.agent._execute_code_safely(self.program.code, self.task_definition, timeout_seconds=1)
        
        self.assertIsNone(results)
        self.assertIsNotNone(error)
        self.assertIn("Execution timed out after 1 seconds", error)

        # Check that 'docker run' was called
        run_call = mock_create_subprocess_exec.call_args_list[0]
        self.assertIn("docker", run_call[0][0])
        self.assertIn("run", run_call[0][1])
        
        # Check that 'docker stop' was called
        stop_call = mock_create_subprocess_exec.call_args_list[1]
        self.assertIn("docker", stop_call[0][0])
        self.assertIn("stop", stop_call[0][1])
        # Extract container name from the run command and check if it's in stop command
        run_args = run_call[0]
        container_name_arg_index = -1
        for i, arg_val in enumerate(run_args):
            if arg_val == "--name":
                container_name_arg_index = i + 1
                break
        self.assertNotEqual(container_name_arg_index, -1, "--name parameter not found in docker run call")
        expected_container_name = run_args[container_name_arg_index]
        self.assertIn(expected_container_name, stop_call[0])

        # proc.kill() on the original docker run proc should also be called by the agent
        mock_proc_docker_run.kill.assert_called_once()


    @patch('evaluator_agent.agent.EvaluatorAgent._execute_code_safely', new_callable=AsyncMock)
    async def test_evaluate_program_successful_evaluation(self, mock_execute_code_safely):
        # --- Test Case: Successful full evaluation ---
        expected_script_output = {
            "test_outputs": [{"test_case_id": 0, "output": 42, "runtime_ms": 10.0, "status": "success"}],
            "average_runtime_ms": 10.0
        }
        mock_execute_code_safely.return_value = (expected_script_output, None)

        evaluated_program = await self.agent.evaluate_program(self.program, self.task_definition)

        self.assertEqual(evaluated_program.status, "evaluated")
        self.assertEqual(evaluated_program.fitness_scores["correctness"], 1.0)
        self.assertEqual(evaluated_program.fitness_scores["passed_tests"], 1.0)
        self.assertEqual(evaluated_program.fitness_scores["total_tests"], 1.0)
        self.assertEqual(evaluated_program.fitness_scores["runtime_ms"], 10.0)
        self.assertEqual(len(evaluated_program.errors), 0)

    @patch('evaluator_agent.agent.EvaluatorAgent._execute_code_safely', new_callable=AsyncMock)
    async def test_evaluate_program_failed_evaluation_due_to_error(self, mock_execute_code_safely):
        # --- Test Case: Failed full evaluation (script error) ---
        mock_execute_code_safely.return_value = (None, "Script crashed badly")

        evaluated_program = await self.agent.evaluate_program(self.program, self.task_definition)

        self.assertEqual(evaluated_program.status, "failed_evaluation")
        self.assertEqual(evaluated_program.fitness_scores["correctness"], 0.0)
        # passed_tests and total_tests might not be set or be 0 if execution fails before assessment
        self.assertIn("Execution Error at Level 0 ('default_level'): Script crashed badly", evaluated_program.errors)

    @patch('evaluator_agent.agent.EvaluatorAgent._execute_code_safely', new_callable=AsyncMock)
    async def test_evaluate_program_failed_evaluation_due_to_incorrect_output(self, mock_execute_code_safely):
        # --- Test Case: Failed full evaluation (incorrect output) ---
        expected_script_output = {
            "test_outputs": [{"test_case_id": 0, "output": 0, "runtime_ms": 10.0, "status": "success"}], # Output is 0, expected 42
            "average_runtime_ms": 10.0
        }
        mock_execute_code_safely.return_value = (expected_script_output, None)

        evaluated_program = await self.agent.evaluate_program(self.program, self.task_definition)

        self.assertEqual(evaluated_program.status, "failed_evaluation")
        self.assertEqual(evaluated_program.fitness_scores["correctness"], 0.0)
        self.assertEqual(evaluated_program.fitness_scores["passed_tests"], 0.0)
        self.assertEqual(evaluated_program.fitness_scores["total_tests"], 1.0)
        self.assertIn("Failed 1 of 1 tests at Level 0 ('default_level').", evaluated_program.errors)

    @patch('evaluator_agent.agent.EvaluatorAgent._execute_code_safely', new_callable=AsyncMock)
    async def test_evaluate_program_with_validation_function(self, mock_execute_code_safely):
        # --- Test Case: Evaluation with validation function ---
        expected_script_output = {
            "test_outputs": [{"test_case_id": 0, "output": 15, "runtime_ms": 10.0, "status": "success"}],
            "average_runtime_ms": 10.0
        }
        mock_execute_code_safely.return_value = (expected_script_output, None)

        # Create a task definition with a validation function
        task_with_validation = TaskDefinition(
            id="test_task_validation",
            description="Test task with validation function",
            function_name_to_evolve="test_function",
            input_output_examples=[
                {
                    "input": [10],
                    "validation_func": """
def validate(input):
    return input > 10
"""
                }
            ]
        )

        evaluated_program = await self.agent.evaluate_program(self.program, task_with_validation)

        self.assertEqual(evaluated_program.status, "evaluated")
        self.assertEqual(evaluated_program.fitness_scores["correctness"], 1.0)
        self.assertEqual(evaluated_program.fitness_scores["passed_tests"], 1.0)
        self.assertEqual(evaluated_program.fitness_scores["total_tests"], 1.0)
        self.assertEqual(len(evaluated_program.errors), 0)

    @patch('evaluator_agent.agent.EvaluatorAgent._execute_code_safely', new_callable=AsyncMock)
    async def test_evaluate_program_with_failed_validation(self, mock_execute_code_safely):
        # --- Test Case: Failed validation function ---
        expected_script_output = {
            "test_outputs": [{"test_case_id": 0, "output": 5, "runtime_ms": 10.0, "status": "success"}],
            "average_runtime_ms": 10.0
        }
        mock_execute_code_safely.return_value = (expected_script_output, None)

        # Create a task definition with a validation function
        task_with_validation = TaskDefinition(
            id="test_task_validation_fail",
            description="Test task with failing validation function",
            function_name_to_evolve="test_function",
            input_output_examples=[
                {
                    "input": [10],
                    "validation_func": """
def validate(input):
    return input > 10
"""
                }
            ]
        )

        evaluated_program = await self.agent.evaluate_program(self.program, task_with_validation)

        self.assertEqual(evaluated_program.status, "failed_evaluation")
        self.assertEqual(evaluated_program.fitness_scores["correctness"], 0.0)
        self.assertEqual(evaluated_program.fitness_scores["passed_tests"], 0.0)
        self.assertEqual(evaluated_program.fitness_scores["total_tests"], 1.0)
        self.assertIn("Failed 1 of 1 tests at Level 0 ('default_level').", evaluated_program.errors)


if __name__ == '__main__':
    unittest.main()

# Need to patch tempfile.mkdtemp to control the temp directory name for more robust assertions on paths
# and to allow cleanup.
# from unittest.mock import patch
# @patch('tempfile.mkdtemp', return_value='/tmp/fixed_temp_dir_for_test')
# ... in test method ...
# mock_mkdtemp.assert_called_once()
# self.assertTrue(os.path.exists('/tmp/fixed_temp_dir_for_test/temp_script.py'))
# ... in tearDown ...
# if os.path.exists('/tmp/fixed_temp_dir_for_test/temp_script.py'):
#     os.remove('/tmp/fixed_temp_dir_for_test/temp_script.py')
# if os.path.exists('/tmp/fixed_temp_dir_for_test'):
#     os.rmdir('/tmp/fixed_temp_dir_for_test')
# This is important because the agent writes the script to temp_dir + "/temp_script.py"
# and this path is part of the `docker run -v` command.
# For now, the test checks for `-v` and `:/app/user_code` which is a good start.
# Checking the exact source path of the volume mount requires patching tempfile.
