# Code Generator Agent
import asyncio  # Added for retry sleep
import logging
import re  # Added for diff application
from typing import Any, Dict, Optional

from google.api_core.exceptions import (  # For specific error handling
    DeadlineExceeded,
    GoogleAPIError,
    InternalServerError,
)
from openai import AsyncOpenAI

from config import settings
from core.interfaces import CodeGeneratorInterface

logger = logging.getLogger(__name__)

class CodeGeneratorAgent(CodeGeneratorInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pro_provider = settings.CUSTOM_PROVIDERS.get(settings.PRO_PROVIDER)
        self.flash_provider_key = settings.FLASH_PROVIDER
        self.flash_provider = settings.CUSTOM_PROVIDERS.get(settings.FLASH_PROVIDER)
        self.evaluation_provider = settings.CUSTOM_PROVIDERS.get(settings.EVALUATION_PROVIDER)


        for provider in settings.CUSTOM_PROVIDERS:
            if not settings.CUSTOM_PROVIDERS[provider]['api_key']:
                raise ValueError(f"{provider} API_KEY not found in settings. Please set it in your .env file or config.")

        # self.clients = { provider : AsyncOpenAI(api_key=settings.CUSTOM_PROVIDERS[provider]['api_key'], base_url=settings.CUSTOM_PROVIDERS[provider]['base_url']) for provider in settings.CUSTOM_PROVIDERS }
        self.client = AsyncOpenAI(base_url=settings.CUSTOM_PROVIDERS[self.flash_provider_key]['base_url'], api_key=settings.CUSTOM_PROVIDERS[self.flash_provider_key]['api_key'])
        self.generation_config = {
            "temperatur":0.7,
            "top_p":0.9,
            "top_k":40,
            #"max_tokens" : 1000,
        }

        # self.max_retries and self.retry_delay_seconds are not used from instance, settings are used directly

    async def generate_code(self, prompt: str, provider_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code") -> str:
        provider_name = provider_name if provider_name else self.flash_provider_key
        logger.info(f"Attempting to generate code using model: {provider_name}, output_format: {output_format}")

        # Add diff instructions if requested
        if output_format == "diff":
            prompt += '''

Provide your changes as a sequence of diff blocks in the following format:
<<<<<<< SEARCH
# Original code block to be found and replaced
=======
# New code block to replace the original
>>>>>>> REPLACE
Ensure the SEARCH block is an exact segment from the current program.
Describe each change with such a SEARCH/REPLACE block.
Make sure that the changes you propose are consistent with each other.
'''
        
        logger.debug(f"Received prompt for code generation (format: {output_format}):\n--PROMPT START--\n{prompt}\n--PROMPT END--")

        if temperature is not None:
            logger.debug(f"Using temperature override: {temperature}")

        retries = settings.API_MAX_RETRIES
        delay = settings.API_RETRY_DELAY_SECONDS
        
        for attempt in range(retries):
            try:
                logger.debug(f"API Call Attempt {attempt + 1} of {retries} to {provider_name}.")

                client = self.client
                client.api_key = settings.CUSTOM_PROVIDERS[provider_name]['api_key']
                client.base_url = settings.CUSTOM_PROVIDERS[provider_name]['base_url']

                response = await client.chat.completions.create(
                    model=settings.CUSTOM_PROVIDERS[provider_name]['model'],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.generation_config["max_tokens"] if "max_tokens" in self.generation_config else None,
                    temperature=temperature,
                    top_p=self.generation_config["top_p"]
                )

                generated_text = response.choices[0].message.content

                logger.debug(f"Raw response from API:\n--RESPONSE START--\n{generated_text}\n--RESPONSE END--")
                if generated_text is None:
                    logger.error("Received None as generated text from API.")
                    return ""
                if output_format == "code":
                    cleaned_code = self._clean_llm_output(generated_text)
                    logger.debug(f"Cleaned code:\n--CLEANED CODE START--\n{cleaned_code}\n--CLEANED CODE END--")
                    return cleaned_code
                else: # output_format == "diff"
                    logger.debug(f"Returning raw diff text:\n--DIFF TEXT START--\n{generated_text}\n--DIFF TEXT END--")
                    return generated_text # Return raw diff text
            except (InternalServerError, DeadlineExceeded, GoogleAPIError) as e:
                logger.warning(f"API error on attempt {attempt + 1}: {type(e).__name__} - {e}. Retrying in {delay}s...")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2 
                else:
                    logger.error(f"API call failed after {retries} retries for model {provider_name}.")
                    raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during code generation with {provider_name}: {e}", exc_info=True)
                raise

        logger.error(f"Code generation failed for model {provider_name} after all retries.")
        return ""

    def _clean_llm_output(self, raw_code: str) -> str:
        """
        Cleans the raw output from the LLM, typically removing markdown code fences.
        Example: ```python\ncode\n``` -> code
        """
        logger.debug(f"Attempting to clean raw LLM output. Input length: {len(raw_code)}")
        code = raw_code.strip()
        
        if code.startswith("```python") and code.endswith("```"):
            cleaned = code[len("```python"): -len("```")].strip()
            logger.debug("Cleaned Python markdown fences.")
            return cleaned
        elif code.startswith("```") and code.endswith("```"):
            cleaned = code[len("```"): -len("```")].strip()
            logger.debug("Cleaned generic markdown fences.")
            return cleaned
            
        logger.debug("No markdown fences found or standard cleaning applied to the stripped code.")
        return code

    def _apply_diff(self, parent_code: str, diff_text: str) -> str:
        """
        Applies a diff in the AlphaEvolve format to the parent code.
        Diff format:
        <<<<<<< SEARCH
        # Original code block
        =======
        # New code block
        >>>>>>> REPLACE
        """
        logger.info("Attempting to apply diff.")
        logger.debug(f"Parent code length: {len(parent_code)}")
        logger.debug(f"Diff text:\n{diff_text}")

        modified_code = parent_code
        diff_pattern = re.compile(r"<<<<<<< SEARCH\s*?\n(.*?)\n=======\s*?\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
        
        for match in diff_pattern.finditer(diff_text):
            search_block = match.group(1)
            replace_block = match.group(2)
            search_block_normalized = search_block.replace('\r\n', '\n').replace('\r', '\n')
            
            try:
                if search_block_normalized in modified_code:
                    modified_code = modified_code.replace(search_block_normalized, replace_block, 1)
                    logger.debug(f"Applied one diff block. SEARCH:\n{search_block_normalized}\nREPLACE:\n{replace_block}")
                else:
                    logger.warning(f"Diff application: SEARCH block not found in current code state:\n{search_block_normalized}")
            except re.error as e:
                logger.error(f"Regex error during diff application: {e}")
                continue
        
        if modified_code == parent_code and diff_text.strip():
             logger.warning("Diff text was provided, but no changes were applied. Check SEARCH blocks/diff format.")
        elif modified_code != parent_code:
             logger.info("Diff successfully applied, code has been modified.")
        else:
             logger.info("No diff text provided or diff was empty, code unchanged.")
             
        return modified_code

    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", parent_code_for_diff: Optional[str] = None) -> str:
        """
        Generic execution method.
        If output_format is 'diff', it generates a diff and applies it to parent_code_for_diff.
        Otherwise, it generates full code.
        """
        logger.debug(f"CodeGeneratorAgent.execute called. Output format: {output_format}")
        
        generated_output = await self.generate_code(
            prompt=prompt,
            provider_name=model_name,
            temperature=temperature,
            output_format=output_format
        )

        if output_format == "diff":
            if not parent_code_for_diff:
                logger.error("Output format is 'diff' but no parent_code_for_diff provided. Returning raw diff.")
                return generated_output 
            
            if not generated_output.strip():
                 logger.info("Generated diff is empty. Returning parent code.")
                 return parent_code_for_diff

            try:
                logger.info("Applying generated diff to parent code.")
                modified_code = self._apply_diff(parent_code_for_diff, generated_output)
                return modified_code
            except Exception as e:
                logger.error(f"Error applying diff: {e}. Returning raw diff text.", exc_info=True)
                return generated_output
        else: # "code"
            return generated_output

# Example Usage (for testing this agent directly)
if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.DEBUG)
    
    async def test_diff_application():
        agent = CodeGeneratorAgent()
        parent = """Line 1
Line 2 to be replaced
Line 3
Another block
To be changed
End of block
Final line"""

        diff = """Some preamble text from LLM...
<<<<<<< SEARCH
Line 2 to be replaced
=======
Line 2 has been successfully replaced
>>>>>>> REPLACE

Some other text...

<<<<<<< SEARCH
Another block
To be changed
End of block
=======
This
Entire
Block
Is New
>>>>>>> REPLACE
Trailing text..."""
        expected_output = """Line 1
Line 2 has been successfully replaced
Line 3
This
Entire
Block
Is New
Final line"""
        
        print("--- Testing _apply_diff directly ---")
        result = agent._apply_diff(parent, diff)
        print("Result of diff application:")
        print(result)
        assert result.strip() == expected_output.strip(), f"Direct diff application failed.\nExpected:\n{expected_output}\nGot:\n{result}"
        print("_apply_diff test passed.")

        print("\n--- Testing execute with output_format='diff' ---")
        async def mock_generate_code(prompt, model_name, temperature, output_format):
            return diff
        
        agent.generate_code = mock_generate_code 
        
        result_execute_diff = await agent.execute(
            prompt="doesn't matter for this mock", 
            parent_code_for_diff=parent,
            output_format="diff"
        )
        print("Result of execute with diff:")
        print(result_execute_diff)
        assert result_execute_diff.strip() == expected_output.strip(), f"Execute with diff failed.\nExpected:\n{expected_output}\nGot:\n{result_execute_diff}"
        print("Execute with diff test passed.")


    async def test_generation():
        agent = CodeGeneratorAgent()
        
        test_prompt_full_code = "Write a Python function that takes two numbers and returns their sum."
        generated_full_code = await agent.execute(test_prompt_full_code, temperature=0.6, output_format="code")
        print("\n--- Generated Full Code (via execute) ---")
        print(generated_full_code)
        print("----------------------")
        assert "def" in generated_full_code, "Full code generation seems to have failed."

        parent_code_for_llm_diff = '''
def greet(name):
    return f"Hello, {name}!"

def process_data(data):
    # TODO: Implement data processing
    return data * 2 # Simple placeholder
'''
        test_prompt_diff_gen = f'''
Current code:
```python
{parent_code_for_llm_diff}
```
Task: Modify the `process_data` function to add 5 to the result instead of multiplying by 2.
Also, change the greeting in `greet` to "Hi, {name}!!!".
'''
        # Commenting out live LLM call for automated testing in this context
        # generated_diff_and_applied = await agent.execute(
        #     prompt=test_prompt_diff_gen,
        #     temperature=0.5,
        #     output_format="diff",
        #     parent_code_for_diff=parent_code_for_llm_diff
        # )
        # print("\n--- Generated Diff and Applied (Live LLM Call) ---")
        # print(generated_diff_and_applied)
        # print("----------------------")
        # assert "data + 5" in generated_diff_and_applied, "LLM diff for process_data not applied as expected."
        # assert "Hi, name!!!" in generated_diff_and_applied, "LLM diff for greet not applied as expected."
        
        async def mock_generate_empty_diff(prompt, model_name, temperature, output_format):
            return "  \n  " 
        
        original_generate_code = agent.generate_code 
        agent.generate_code = mock_generate_empty_diff
        
        print("\n--- Testing execute with empty diff from LLM ---")
        result_empty_diff = await agent.execute(
            prompt="doesn't matter",
            parent_code_for_diff=parent_code_for_llm_diff,
            output_format="diff"
        )
        assert result_empty_diff == parent_code_for_llm_diff, "Empty diff should return parent code."
        print("Execute with empty diff test passed.")
        agent.generate_code = original_generate_code

    async def main_tests():
        await test_diff_application()
        # await test_generation() # Uncomment to run live LLM tests for generate_code
        print("\nAll selected local tests in CodeGeneratorAgent passed.")

    asyncio.run(main_tests())
