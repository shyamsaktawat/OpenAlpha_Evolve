import logging
import re
from typing import Optional, Dict, Any
import asyncio

from litellm import acompletion
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    Timeout
)

from config import settings
from core.interfaces import CodeGeneratorInterface

logger = logging.getLogger(__name__)

class CodeGeneratorAgent(CodeGeneratorInterface):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Estes são agora padrões que podem ser substituídos pelas configurações da UI
        self.model_name = settings.LITELLM_DEFAULT_MODEL
        self.generation_config = {
            "temperature": float(settings.LITELLM_TEMPERATURE),
            "top_p": float(settings.LITELLM_TOP_P),
            "top_k": int(settings.LITELLM_TOP_K),
            "max_tokens": int(settings.LITELLM_MAX_TOKENS),
        }
        self.litellm_extra_params = {
            "base_url": settings.LITELLM_DEFAULT_BASE_URL,
        }
        logger.info(f"CodeGeneratorAgent initialized with default model: {self.model_name}")

    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", litellm_extra_params: Optional[Dict[str, Any]] = None) -> str:
        # MODIFICADO: Prioriza o model_name passado durante a execução (da UI)
        effective_model_name = model_name if model_name else self.model_name
        
        # MODIFICADO: Lida de forma inteligente com parâmetros extras para o Ollama
        effective_litellm_params = self.litellm_extra_params.copy()
        if litellm_extra_params:
            effective_litellm_params.update(litellm_extra_params)
            
        # Se estiver usando um modelo ollama, garanta que a base_url esteja definida corretamente
        if effective_model_name.startswith("ollama/"):
            if "base_url" not in effective_litellm_params or not effective_litellm_params["base_url"]:
                # Padrão para Ollama local se não for fornecido
                effective_litellm_params["base_url"] = "http://localhost:11434"
            logger.info(f"Ollama model detected. Using base_url: {effective_litellm_params.get('base_url')}")

        logger.info(f"Attempting to generate code using model: {effective_model_name}, output_format: {output_format}")
        
        if output_format == "diff":
            prompt += '''

I need you to provide your changes as a sequence of diff blocks in the following format:

# New code block to replace the original

IMPORTANT DIFF GUIDELINES:
1. The SEARCH block MUST be an EXACT copy of code from the original - match whitespace, indentation, and line breaks precisely
2. Each SEARCH block should be large enough (3-5 lines minimum) to uniquely identify where the change should be made
3. Include context around the specific line(s) you want to change
4. Make multiple separate diff blocks if you need to change different parts of the code
5. For each diff, the SEARCH and REPLACE blocks must be complete, valid code segments
6. Pay special attention to matching the exact original indentation of the code in your SEARCH block, as this is crucial for correct application in environments sensitive to indentation (like Python).

Example of a good diff:
def calculate_sum(numbers):
    if not numbers:
        return 0
    result = 0
    for num in numbers:
        result += num
    return result

Make sure your diff can be applied correctly!
'''
        
        logger.debug(f"Received prompt for code generation (format: {output_format}):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        
        current_generation_config = self.generation_config.copy()
        if temperature is not None:
            current_generation_config["temperature"] = temperature
            logger.debug(f"Using temperature override: {temperature}")

        retries = settings.API_MAX_RETRIES
        delay = settings.API_RETRY_DELAY_SECONDS
        
        for attempt in range(retries):
            try:
                logger.debug(f"API Call Attempt {attempt + 1} of {retries} to {effective_model_name}.")
                response = await acompletion(
                    model=effective_model_name,
                    messages=[{"role": "user", "content": prompt + " \nothink"}],
                    **(current_generation_config or {}),
                    **(effective_litellm_params or {})
                )
                
                if not response.choices or not response.choices[0].message.content:
                    logger.warning("LLM API returned no choices or empty content.")
                    return ""

                generated_text = response.choices[0].message.content
                logger.debug(f"Raw response from LLM API:\n--RESPONSE START--\n{generated_text}\n--RESPONSE END--")
                
                if output_format == "code":
                    cleaned_code = self._clean_llm_output(generated_text)
                    logger.debug(f"Cleaned code:\n--CLEANED CODE START--\n{cleaned_code}\n--CLEANED CODE END--")
                    return cleaned_code
                else:                           
                    logger.debug(f"Returning raw diff text:\n--DIFF TEXT START--\n{generated_text}\n--DIFF TEXT END--")
                    return generated_text                        
            except (APIError, InternalServerError, Timeout, RateLimitError, AuthenticationError, BadRequestError) as e:
                logger.warning(f"LLM API error on attempt {attempt + 1}: {type(e).__name__} - {e}. Retrying in {delay}s...")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 1.5 
                else:
                    logger.error(f"LLM API call failed after {retries} retries for model {effective_model_name}.")
                    raise
            except Exception as e:
                logger.error(f"An unexpected error occurred during code generation with {effective_model_name}: {e}", exc_info=True)
                raise
        
        logger.error(f"Code generation failed for model {effective_model_name} after all retries.")
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
        
        Uses fuzzy matching to handle slight variations in whitespace and indentation.
        """
        logger.info("Attempting to apply diff.")
        logger.debug(f"Parent code length: {len(parent_code)}")
        logger.debug(f"Diff text:\n{diff_text}")

        modified_code = parent_code
        diff_pattern = re.compile(r"<<<<<<< SEARCH\s*?\n(.*?)\n=======\s*?\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
        
        replacements_made = []
        
        for match in diff_pattern.finditer(diff_text):
            search_block = match.group(1)
            replace_block = match.group(2)
            
            search_block_normalized = search_block.replace('\r\n', '\n').replace('\r', '\n').strip()
            
            try:
                if search_block_normalized in modified_code:
                    logger.debug(f"Found exact match for SEARCH block")
                    modified_code = modified_code.replace(search_block_normalized, replace_block, 1)
                    logger.debug(f"Applied one diff block. SEARCH:\n{search_block_normalized}\nREPLACE:\n{replace_block}")
                else:
                    normalized_search = re.sub(r'\s+', ' ', search_block_normalized)
                    normalized_code = re.sub(r'\s+', ' ', modified_code)
                    
                    if normalized_search in normalized_code:
                        logger.debug(f"Found match after whitespace normalization")
                        start_pos = normalized_code.find(normalized_search)
                        
                        original_pos = 0
                        norm_pos = 0
                        
                        while norm_pos < start_pos and original_pos < len(modified_code):
                            if not modified_code[original_pos].isspace() or (
                                original_pos > 0 and 
                                modified_code[original_pos].isspace() and 
                                not modified_code[original_pos-1].isspace()
                            ):
                                norm_pos += 1
                            original_pos += 1
                        
                        end_pos = original_pos
                        remaining_chars = len(normalized_search)
                        
                        while remaining_chars > 0 and end_pos < len(modified_code):
                            if not modified_code[end_pos].isspace() or (
                                end_pos > 0 and 
                                modified_code[end_pos].isspace() and 
                                not modified_code[end_pos-1].isspace()
                            ):
                                remaining_chars -= 1
                            end_pos += 1
                        
                        overlap = False
                        for start, end in replacements_made:
                            if (start <= original_pos <= end) or (start <= end_pos <= end):
                                overlap = True
                                break
                        
                        if not overlap:
                            actual_segment = modified_code[original_pos:end_pos]
                            logger.debug(f"Replacing segment:\n{actual_segment}\nWith:\n{replace_block}")
                            
                            modified_code = modified_code[:original_pos] + replace_block + modified_code[end_pos:]
                            
                            replacements_made.append((original_pos, original_pos + len(replace_block)))
                        else:
                            logger.warning(f"Diff application: Skipping overlapping replacement")
                    else:
                        search_lines = search_block_normalized.splitlines()
                        parent_lines = modified_code.splitlines()
                        
                        if len(search_lines) >= 3:
                            first_line = search_lines[0].strip()
                            last_line = search_lines[-1].strip()
                            
                            for i, line in enumerate(parent_lines):
                                if first_line in line.strip() and i + len(search_lines) <= len(parent_lines):
                                    if last_line in parent_lines[i + len(search_lines) - 1].strip():
                                        matched_segment = '\n'.join(parent_lines[i:i + len(search_lines)])
                                        
                                        modified_code = '\n'.join(
                                            parent_lines[:i] + 
                                            replace_block.splitlines() + 
                                            parent_lines[i + len(search_lines):]
                                        )
                                        logger.debug(f"Applied line-by-line match. SEARCH:\n{matched_segment}\nREPLACE:\n{replace_block}")
                                        break
                            else:
                                logger.warning(f"Diff application: SEARCH block not found even with line-by-line search:\n{search_block_normalized}")
                        else:
                            logger.warning(f"Diff application: SEARCH block not found in current code state:\n{search_block_normalized}")
            except re.error as e:
                logger.error(f"Regex error during diff application: {e}")
                continue
            except Exception as e:
                logger.error(f"Error during diff application: {e}", exc_info=True)
                continue
        
        if modified_code == parent_code and diff_text.strip():
             logger.warning("Diff text was provided, but no changes were applied. Check SEARCH blocks/diff format.")
        elif modified_code != parent_code:
             logger.info("Diff successfully applied, code has been modified.")
        else:
             logger.info("No diff text provided or diff was empty, code unchanged.")
             
        return modified_code

    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", parent_code_for_diff: Optional[str] = None, litellm_extra_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generic execution method.
        If output_format is 'diff', it generates a diff and applies it to parent_code_for_diff.
        Otherwise, it generates full code.
        """
        logger.debug(f"CodeGeneratorAgent.execute called. Output format: {output_format}")
        
        generated_output = await self.generate_code(
            prompt=prompt, 
            model_name=model_name, 
            temperature=temperature,
            output_format=output_format,
            litellm_extra_params=litellm_extra_params
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
        else:         
            return generated_output

if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.DEBUG)