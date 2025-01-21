import os
from abc import ABC
from typing import List

from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

from src.configs import LLMConfig

from dotenv import load_dotenv

load_dotenv()


class BaseAgent(ABC):
    def __init__(self, config: LLMConfig):
        super().__init__()

        self.config = config
        self.client = None
    
    def generate(self, messages: List[str], is_stream: bool = False):
        response = self.client.stream(messages) if is_stream \
            else self.client.invoke(messages)
        return response


class OpenAIAgent(BaseAgent):
    def __init__(self, config: LLMConfig):
        super().__init__(config)

        self.client = ChatOpenAI(
            model=self.config.model, 
            api_key=os.getenv("API_KEY"), 
            temperature=self.config.temperature,
        )


class OllamaAgent(BaseAgent):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        self.client = ChatOllama(
            model=self.config.model, 
            temperature=self.config.temperature,
        )