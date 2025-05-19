#!/usr/bin/env python3
"""
Configuration for OpenAI API integration.
This file reads the OpenAI API key and model name from environment variables.
"""
import os

# OpenAI API Key (set OPENAI_API_KEY in your .env or environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4")

# You can add additional OpenAI-specific settings below
# e.g., request timeout, temperature defaults, etc. 