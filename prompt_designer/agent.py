                        
from typing import Optional, Dict, Any
import logging

from core.interfaces import PromptDesignerInterface, Program, TaskDefinition, BaseAgent

logger = logging.getLogger(__name__)

class PromptDesignerAgent(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):
        super().__init__()
        self.task_definition = task_definition
        logger.info(f"PromptDesignerAgent initialized for task: {self.task_definition.id}")

    def design_initial_prompt(self) -> str:
        logger.info(f"Designing initial prompt for task: {self.task_definition.id}")
                                                           
        expert_knowledge_section = ""
        if self.task_definition.expert_knowledge:
            expert_knowledge_section = f"Relevant Expert Knowledge or Context:\\n{self.task_definition.expert_knowledge}\\n\\n"

        prompt = (
            f"You are an expert Python programmer. Your task is to write a Python function based on the following specifications.\\n\\n"
            f"Task Description: {self.task_definition.description}\\n\\n"
            f"{expert_knowledge_section}"
            f"Function to Implement: `{self.task_definition.function_name_to_evolve}`\\n\\n"
            f"Input/Output Examples:\n"
                                         
            f"{self._format_input_output_examples()}\n\n"
            f"Evaluation Criteria: {self.task_definition.evaluation_criteria}\n\n"
            f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use any other external libraries or packages.\n\n"
            f"Your Response Format:\n"
            f"Please provide *only* the complete Python code for the function `{self.task_definition.function_name_to_evolve}`. "
            f"The code should be self-contained or rely only on the allowed imports. "
            f"Do not include any surrounding text, explanations, comments outside the function, or markdown code fences (like ```python or ```)."
        )
        logger.debug(f"Designed initial prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _format_input_output_examples(self) -> str:
        """Format input/output examples for the prompt."""
        examples = []
        
        # YAML format with tests
        if self.task_definition.tests:
            for test_group in self.task_definition.tests:
                for test_case in test_group['test_cases']:
                    if 'output' in test_case:
                        examples.append(f"Input: {test_case['input']}\nOutput: {test_case['output']}")
                    elif 'validation_func' in test_case:
                        examples.append(f"Input: {test_case['input']}\nValidation: {test_case['validation_func']}")
        
        # legacy format with input_output_examples
        elif self.task_definition.input_output_examples:
            for example in self.task_definition.input_output_examples:
                input_str = str(example.get('input'))
                output_str = str(example.get('output'))
                examples.append(f"Input: {input_str}\nOutput: {output_str}")
        
        if not examples:
            return "No input/output examples provided."
        
        return "\n\n".join(examples)

    def _format_evaluation_feedback(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]]) -> str:
        if not evaluation_feedback:
            return "No detailed evaluation feedback is available for the previous version of this code. Attempt a general improvement or refinement."

        correctness = evaluation_feedback.get("correctness_score", None)
        runtime = evaluation_feedback.get("runtime_ms", None)
        errors = evaluation_feedback.get("errors", [])                          
                                                                                               
        stderr = evaluation_feedback.get("stderr", None)

        feedback_parts = []
        if correctness is not None:
            feedback_parts.append(f"- Correctness Score: {correctness*100:.2f}%")
        if runtime is not None:
            feedback_parts.append(f"- Runtime: {runtime:.2f} ms")
        
        if errors:
            error_messages = "\n".join([f"  - {e}" for e in errors])
            feedback_parts.append(f"- Errors Encountered During Evaluation:\n{error_messages}")
        elif stderr:
            feedback_parts.append(f"- Standard Error Output During Execution:\n{stderr}")
        elif correctness is not None and correctness < 1.0:
            feedback_parts.append("- The code did not achieve 100% correctness but produced no explicit errors or stderr. Review logic for test case failures.")
        elif correctness == 1.0:
            feedback_parts.append("- The code achieved 100% correctness. Consider optimizing for efficiency or exploring alternative correct solutions.")
        
        if not feedback_parts:
             return "The previous version was evaluated, but no specific feedback details were captured. Try a general improvement."

        return "Summary of the previous version's evaluation:\n" + "\n".join(feedback_parts)

    def design_mutation_prompt(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Designing mutation prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Parent program code (to be mutated):\\n{program.code}")
        
        expert_knowledge_section = ""
        if self.task_definition.expert_knowledge:
            expert_knowledge_section = f"Relevant Expert Knowledge or Context (applies to the overall task):\\n{self.task_definition.expert_knowledge}\\n\\n"

        feedback_summary = self._format_evaluation_feedback(program, evaluation_feedback)
        logger.debug(f"Formatted evaluation feedback for prompt:\n{feedback_summary}")

        diff_instructions = (
            "Your Response Format:\n"
            "Propose improvements to the 'Current Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Current Code'. Do not paraphrase or shorten it."
            "- If you are adding new code where nothing existed, the SEARCH block can be a comment indicating the location, or an adjacent existing line."
            "- If you are deleting code, the REPLACE block should be empty."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to improve an existing Python function based on its previous performance and the overall goal.\\n\\n"
            f"Overall Task Description: {self.task_definition.description}\\n\\n"
            f"{expert_knowledge_section}"
            f"Function to Improve: `{self.task_definition.function_name_to_evolve}`\\n\\n"
            f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Current Code (Version from Generation {program.generation}):\n"
            f"```python\n{program.code}\n```\n\n"
            f"Evaluation Feedback on the 'Current Code':\n{feedback_summary}\n\n"
            f"Your Improvement Goal:\n"
            f"Based on the task, the 'Current Code', and its 'Evaluation Feedback', your goal is to propose modifications to improve the function `{self.task_definition.function_name_to_evolve}`. "
            f"Prioritize fixing any errors or correctness issues. If correct, focus on improving efficiency or exploring alternative robust logic. "
            f"Consider the original evaluation criteria: {self.task_definition.evaluation_criteria}\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed mutation prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def design_bug_fix_prompt(self, program: Program, error_message: str, execution_output: Optional[str] = None) -> str:
        logger.info(f"Designing bug-fix prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Buggy program code:\\n{program.code}")
        
        expert_knowledge_section = ""
        if self.task_definition.expert_knowledge:
            expert_knowledge_section = f"Relevant Expert Knowledge or Context (applies to the overall task):\\n{self.task_definition.expert_knowledge}\\n\\n"

        logger.debug(f"Primary error message: {error_message}")
        if execution_output:
            logger.debug(f"Additional execution output (stdout/stderr): {execution_output}")

        output_segment = f"Execution Output (stdout/stderr that might be relevant):\n{execution_output}\n" if execution_output else "No detailed execution output was captured beyond the error message itself.\n"
        
        diff_instructions = (
            "Your Response Format:\n"
            "Propose fixes to the 'Buggy Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Buggy Code'."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to fix a bug in an existing Python function.\\n\\n"
            f"Overall Task Description: {self.task_definition.description}\\n\\n"
            f"{expert_knowledge_section}"
            f"Function to Fix: `{self.task_definition.function_name_to_evolve}`\\n\\n"
            f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Buggy Code (Version from Generation {program.generation}):\n"
            f"```python\n{program.code}\n```\n\n"
            f"Error Encountered: {error_message}\n"
            f"{output_segment}\n"
            f"Your Goal:\n"
            f"Analyze the 'Buggy Code', the 'Error Encountered', and any 'Execution Output' to identify and fix the bug(s). "
            f"The corrected function must adhere to the overall task description and allowed imports.\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed bug-fix prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PromptDesignerAgent.execute() is not the primary way to use this agent. Call specific design methods.")

                
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    sample_task_def = TaskDefinition(
        id="task_001_designer_test",
        description="Create a Python function `sum_list(numbers)` that returns the sum of a list of integers. Handle empty lists by returning 0.",
        function_name_to_evolve="sum_list",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0}
        ],
        evaluation_criteria="Ensure correctness for all cases, including empty lists.",
        allowed_imports=["math"]
    )
    designer = PromptDesignerAgent(task_definition=sample_task_def)

    print("--- Initial Prompt ---")
    initial_prompt = designer.design_initial_prompt()
    print(initial_prompt)

    sample_program_mutation = Program(
        id="prog_mut_001",
        code="def sum_list(numbers):\n  # Slightly off logic\n  s = 0\n  for n in numbers:\n    s += n\n  return s if numbers else 1",                     
        fitness_scores={"correctness_score": 0.5, "runtime_ms": 5.0},
        generation=1,
        errors=["Test case failed: Input [], Expected 0, Got 1"],
        status="evaluated"
    )
    mutation_feedback = {
        "correctness_score": sample_program_mutation.fitness_scores["correctness_score"],
        "runtime_ms": sample_program_mutation.fitness_scores["runtime_ms"],
        "errors": sample_program_mutation.errors,
        "stderr": None
    }
    print("\n--- Mutation Prompt (Requesting Diff) ---")
    mutation_prompt = designer.design_mutation_prompt(sample_program_mutation, evaluation_feedback=mutation_feedback)
    print(mutation_prompt)

    sample_program_buggy = Program(
        id="prog_bug_002",
        code="def sum_list(numbers):\n  # Buggy implementation causing TypeError\n  if not numbers:\n    return 0\n  return sum(numbers) + \"oops\"",
        fitness_scores={"correctness_score": 0.0, "runtime_ms": 2.0},
        generation=2,
        errors=["TypeError: unsupported operand type(s) for +: 'int' and 'str'"],
        status="evaluated"
    )
    print("\n--- Bug-Fix Prompt (Requesting Diff) ---")
    bug_fix_prompt = designer.design_bug_fix_prompt(sample_program_buggy, error_message=sample_program_buggy.errors[0], execution_output="TypeError occurred during summation.")
    print(bug_fix_prompt) 