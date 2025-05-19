                      
import os
from dotenv import load_dotenv

                                           
load_dotenv()

                             
GEMINI_API_KEY = os.getenv("")

                                                                  
                                                    
if not GEMINI_API_KEY:
                       
                                                 
                                                       
                                                                                                   
                                                                             
                                                 
    print("Warning: GEMINI_API_KEY not found in .env or environment. Using a NON-FUNCTIONAL placeholder. Please create a .env file with your valid API key.")
    GEMINI_API_KEY = ""                      

                         
GEMINI_PRO_MODEL_NAME = "gemini-2.5-flash-preview-04-17"                             
GEMINI_FLASH_MODEL_NAME = "gemini-2.5-flash-preview-04-17"                          
GEMINI_EVALUATION_MODEL = "gemini-2.5-flash-preview-04-17"                             

                                    
POPULATION_SIZE = 5                                            
GENERATIONS = 2                                                   
ELITISM_COUNT = 1                                                                      
MUTATION_RATE = 0.7                                          
CROSSOVER_RATE = 0.2                                                                          

                     
EVALUATION_TIMEOUT_SECONDS = 800                                                   

                                                            
DATABASE_TYPE = "in_memory"                                          
DATABASE_PATH = "program_database.json"                         

                    
LOG_LEVEL = "INFO"                                        
LOG_FILE = "alpha_evolve.log"

                      
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
        return GEMINI_PRO_MODEL_NAME
    elif model_type == "flash":
        return GEMINI_FLASH_MODEL_NAME
    return GEMINI_FLASH_MODEL_NAME                   

                                 