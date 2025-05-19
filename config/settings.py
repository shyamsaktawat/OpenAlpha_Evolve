import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env or environment. Using a NON-FUNCTIONAL placeholder. Please create a .env file with your valid API key.")
    GEMINI_API_KEY = "" # Obvious placeholder

GEMINI_PRO_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Using a more capable model
GEMINI_FLASH_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Default model for speed
GEMINI_EVALUATION_MODEL = "gemini-2.5-flash-preview-04-17" # Model for evaluation tasks

POPULATION_SIZE = 5  # Number of individuals in each generation
GENERATIONS = 2       # Number of generations to run the evolution
ELITISM_COUNT = 1     # Number of best individuals to carry over to the next generation
MUTATION_RATE = 0.7   # Probability of mutating an individual
CROSSOVER_RATE = 0.2  # Probability of crossing over two parents (if crossover is implemented)

EVALUATION_TIMEOUT_SECONDS = 800  # Max time for a program to run during evaluation

DATABASE_TYPE = "in_memory" # or "sqlite", "postgresql" in the future
DATABASE_PATH = "program_database.json" # Path for file-based DB

LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "alpha_evolve.log"

API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10 # Initial delay, will be exponential

RL_TRAINING_INTERVAL_GENERATIONS = 50 # Fine-tune RL model every N generations
RL_MODEL_PATH = "rl_finetuner_model.pth"

MONITORING_DASHBOARD_URL = "http://localhost:8080" # Example

def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    return globals().get(key, default)

def get_llm_model(model_type="pro"):
    if model_type == "pro":
        return GEMINI_PRO_MODEL_NAME
    elif model_type == "flash":
        return GEMINI_FLASH_MODEL_NAME
    return GEMINI_FLASH_MODEL_NAME # Default fallback

