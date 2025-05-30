import os
import sys # Import sys module
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
FLASH_API_KEY = os.getenv("FLASH_API_KEY")
if not FLASH_API_KEY:
    sys.stderr.write("Error: FLASH_API_KEY not found in .env or environment.\n")
    sys.exit(1)

FLASH_BASE_URL = os.getenv("FLASH_BASE_URL", None)
FLASH_MODEL = os.getenv("FLASH_MODEL")

PRO_API_KEY = os.getenv("PRO_API_KEY")
if not PRO_API_KEY:
    sys.stderr.write("Error: PRO_API_KEY not found in .env or environment.\n")
    sys.exit(1)

PRO_BASE_URL = os.getenv("PRO_BASE_URL", None)
PRO_MODEL = os.getenv("PRO_MODEL")

EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY")
if not EVALUATION_API_KEY:
    sys.stderr.write("Error: EVALUATION_API_KEY not found in .env or environment.\n")
    sys.exit(1)

EVALUATION_BASE_URL = os.getenv("EVALUATION_BASE_URL", None)
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL")

# LiteLLM Configuration
# Read from environment or use default values, then cast to correct type
raw_max_tokens = os.getenv("LITELLM_MAX_TOKENS")
LITELLM_MAX_TOKENS = int(raw_max_tokens) if raw_max_tokens else 4096

raw_temperature = os.getenv("LITELLM_TEMPERATURE")
LITELLM_TEMPERATURE = float(raw_temperature) if raw_temperature else 0.7

raw_top_p = os.getenv("LITELLM_TOP_P")
LITELLM_TOP_P = float(raw_top_p) if raw_top_p else 1.0

raw_top_k = os.getenv("LITELLM_TOP_K")
LITELLM_TOP_K = int(raw_top_k) if raw_top_k else 40

# Evolutionary Algorithm Settings
POPULATION_SIZE = 5
GENERATIONS = 2
ELITISM_COUNT = 1
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.2

# Island Model Settings
NUM_ISLANDS = 4  # Number of subpopulations
MIGRATION_INTERVAL = 4  # Number of generations between migrations
ISLAND_POPULATION_SIZE = POPULATION_SIZE // NUM_ISLANDS  # Programs per island
MIN_ISLAND_SIZE = 2  # Minimum number of programs per island
MIGRATION_RATE = 0.2  # Rate at which programs migrate between islands

# Debug Settings
DEBUG = os.getenv("DEBUG", False)
EVALUATION_TIMEOUT_SECONDS = 800

DATABASE_TYPE = "in_memory"
DATABASE_PATH = "program_database.json"

# Logging Configuration
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"
LOG_FILE = "alpha_evolve.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

API_MAX_RETRIES = 5
API_RETRY_DELAY_SECONDS = 10

RL_TRAINING_INTERVAL_GENERATIONS = 50
RL_MODEL_PATH = "rl_finetuner_model.pth"

MONITORING_DASHBOARD_URL = "http://localhost:8080"

def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    return globals().get(key, default)

def get_llm_model(model_type="pro"):
    if model_type == "pro":
        return PRO_MODEL
    elif model_type == "flash":
        return FLASH_MODEL
    elif model_type == "evaluation":
        return EVALUATION_MODEL
    return FLASH_MODEL  # Default fallback for unknown types
