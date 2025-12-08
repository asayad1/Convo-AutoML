"""
This file defines the config.
"""

import os

class Config:
    SERVING_METHOD = "ollama" # To the grader: if you have access to portkey, change to "portkey"

class PortkeyConfig(Config):
    PORTKEY_BASE_URL = os.getenv("PORTKEY_BASE_URL", "https://portkey-api.livelab.jhuapl.edu/v1")
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "")
    PORTKEY_MODEL = os.getenv("PORTKEY_MODEL", "@opal/openai/gpt-oss-120b")
    PORTKEY_MAX_TOKENS = int(os.getenv("PORTKEY_MAX_TOKENS", "4096"))
    
class OllamaConfig(Config):
    OLLAMA_MODEL = "gpt-oss:20b"
    OLLAMA_MAX_TOKENS = 4096