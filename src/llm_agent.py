import os
from typing import List

from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

from src.configs import LLMConfig

from dotenv import load_dotenv

load_dotenv()


class BaseAgent:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
    
    def generate_response(self, messages: List[str], is_stream: bool = False):
        response = self.client.stream(messages) if is_stream \
            else self.client.invoke(messages)
        return response


class OpenAIAgent(BaseAgent):
    def __init__(self):
        self.client = ChatOpenAI(
            model=self.config.model, 
            api_key=os.getenv("API_KEY"), 
            temperature=self.config.temperature,
        )


class OllamaAgent(BaseAgent):
    def __init__(self):
        self.client = ChatOllama(
            model=self.config.model, 
            temperature=self.config.temperature,
        )