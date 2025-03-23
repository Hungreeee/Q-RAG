import os
import httpx
from typing import List, Tuple

from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama

from deepeval.models import DeepEvalBaseLLM

from src.configs import LLMConfig

from dotenv import load_dotenv

load_dotenv()


def update_base_url(request: httpx.Request, model: str):
    if request.url.path == "/chat/completions":
        if model == "gpt-4o":
            request.url = request.url.copy_with(path="/v1/openai/gpt4o/chat/completions")
        elif model == "gpt-35-turbo":
            request.url = request.url.copy_with(path="/v1/chat/")
        elif model == "gpt-4-turbo":
            request.url = request.url.copy_with(path="/v1/openai/gpt4-turbo/chat/completions")
        elif model == "gpt-4-8k":
            request.url = request.url.copy_with(path="/v1/chat/gpt4-8k")
        else:
            raise Exception(f"Model {model} is not currently supported.")
        

class BaseAgent:
    def __init__(self, config: LLMConfig = LLMConfig.default()):
        self.config = config
        self.client = None
    
    def generate(self, messages: List[Tuple[str]], is_stream: bool = False):
        response = self.client.stream(messages) if is_stream \
            else self.client.invoke(messages)
        return response


class OpenAIAgent(BaseAgent):
    def __init__(self, config: LLMConfig = LLMConfig.default()):
        super().__init__(config)

        self.client = ChatOpenAI(
            model=self.config.model, 
            api_key=os.getenv("OPENAI_API_KEY"), 
            temperature=self.config.temperature,
        )


class OllamaAgent(BaseAgent):
    def __init__(
        self, 
        config: LLMConfig = LLMConfig.default(), 
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(config)
        
        self.client = ChatOllama(
            base_url=base_url,
            model=self.config.model, 
            temperature=self.config.temperature,
        )


class AzureAIAgent(BaseAgent):
    def __init__(
        self, 
        config: LLMConfig = LLMConfig.default(),
        base_url: str = "https://aalto-openai-apigw.azure-api.net",
    ):
        super().__init__(config)

        self.client = ChatOpenAI(
            default_headers={
                "Ocp-Apim-Subscription-Key": os.getenv("AALTO_OPENAI_API_KEY")
            },
            base_url=base_url,
            api_key=None,
            http_client=httpx.Client(
            event_hooks={
                "request": [lambda request: update_base_url(request, model=self.config.model)],
            }),
            temperature=self.config.temperature,
        )


class LLMJudge(DeepEvalBaseLLM):
    def __init__(self, 
        config: LLMConfig = LLMConfig.default(),
        base_url: str = "https://aalto-openai-apigw.azure-api.net"
    ):
        self.config = config

        self.client = ChatOpenAI(
            default_headers={
                "Ocp-Apim-Subscription-Key": os.getenv("AALTO_OPENAI_API_KEY")
            },
            base_url=base_url,
            api_key=None,
            http_client=httpx.Client(
            event_hooks={
                "request": [lambda request: update_base_url(request, model=self.config.model)],
            }),
            temperature=self.config.temperature,
        )

    def load_model(self):
        return self.client
    
    def generate(self, prompt: str):
        response = self.client.invoke(prompt)
        return response.content
    
    async def a_generate(self, prompt: str):
        return self.generate(prompt)
    
    def get_model_name(self):
        return self.config.model