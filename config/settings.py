# Configuration files 
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Main API key for Gemini (fallback provider)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Provider identifiers
PRO_KEY = "PRO"
FLASH_KEY = "FLASH"
EVALUATION_KEY = "EVALUATION"

# LLM Model Configuration
GEMINI_PRO_MODEL_NAME = os.getenv("GEMINI_PRO_MODEL_NAME", "gemini-2.5-flash-preview-04-17")
GEMINI_FLASH_MODEL_NAME = os.getenv("GEMINI_FLASH_MODEL_NAME", "gemini-2.5-flash-preview-04-17")
GEMINI_EVALUATION_MODEL = os.getenv("GEMINI_EVALUATION_MODEL", "gemini-2.5-flash-preview-04-17")

# Custom provider settings for code generation
CUSTOM_PROVIDERS = {
    PRO_KEY: {
        "base_url": os.getenv("PRO_BASE_URL", ""),
        "api_key": os.getenv("PRO_API_KEY", GEMINI_API_KEY),
        "model": os.getenv("PRO_MODEL", GEMINI_PRO_MODEL_NAME)
    },
    FLASH_KEY: {
        "base_url": os.getenv("FLASH_BASE_URL", ""),
        "api_key": os.getenv("FLASH_API_KEY", GEMINI_API_KEY),
        "model": os.getenv("FLASH_MODEL", GEMINI_FLASH_MODEL_NAME)
    },
    EVALUATION_KEY: {
        "base_url": os.getenv("EVALUATION_BASE_URL", ""),
        "api_key": os.getenv("EVALUATION_API_KEY", GEMINI_API_KEY),
        "model": os.getenv("EVALUATION_MODEL", GEMINI_EVALUATION_MODEL)
    }
}

# Evolutionary Parameters (examples)
POPULATION_SIZE = 5  # Number of individuals in each generation
GENERATIONS = 2       # Number of generations to run the evolution
ELITISM_COUNT = 1     # Number of best individuals to carry over to the next generation
MUTATION_RATE = 0.7   # Probability of mutating an individual
CROSSOVER_RATE = 0.2  # Probability of crossing over two parents (if crossover is implemented)

# Evaluation settings
EVALUATION_TIMEOUT_SECONDS = 800  # Max time for a program to run during evaluation

# Database settings (using a simple in-memory store for now)
DATABASE_TYPE = "in_memory" # or "sqlite", "postgresql" in the future
DATABASE_PATH = "program_database.json" # Path for file-based DB

# Logging Parameters
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "alpha_evolve.log"

# API Retry Parameters
API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10 # Initial delay, will be exponential

# Placeholder for RL Fine-Tuning (if implemented)
RL_TRAINING_INTERVAL_GENERATIONS = 50 # Fine-tune RL model every N generations
RL_MODEL_PATH = "rl_finetuner_model.pth"

# Monitoring (if implemented)
MONITORING_DASHBOARD_URL = "http://localhost:8080" # Example

# Fallback for development: warn if main API key isn't set
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env or environment. Some providers may not work without a valid key.")
    GEMINI_API_KEY = None

# --- Helper function to get a specific setting ---
def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    # Prioritize environment variables for some settings if needed
    # For example: return os.getenv(key, globals().get(key, default))
    return globals().get(key, default)

# Example of how to get a model, perhaps with fallback logic (not strictly necessary with current direct assignments)
def get_llm_model(model_type="pro"):
    if model_type == "pro":
        return GEMINI_PRO_MODEL_NAME
    elif model_type == "flash":
        return GEMINI_FLASH_MODEL_NAME
    return GEMINI_FLASH_MODEL_NAME # Default fallback

# Add other global settings here 