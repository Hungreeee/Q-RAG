from abc import ABC, abstractmethod

from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_ollama import OllamaLLM


class BaseClient(ABC):
    def __init__(
        self,
        model: str,
        context_size: int,
        max_output_tokens: int,
    ) -> None:
        self.model = model
        self.context_size = context_size
        self.output_token_estimate = max_output_tokens


    def max_tokens(self) -> int:
        return self.context_size

    @abstractmethod
    def run(
        self,
        message: PromptValue,
        logging: bool = False
    ) -> str:
        pass
    

class OpenAIClient(BaseClient):
    def __init__(
        self,
        model: str,
        context_size: int,
        max_output_tokens: int,
        use_azure_openai: bool = False,
        **krawgs
    ):
        super.__init__(model, context_size, max_output_tokens)

        self.client = ChatOpenAI(
            model=self.model,
            **krawgs
        ) if not use_azure_openai else AzureChatOpenAI(
            model=self.model,
            **krawgs
        )
    

    def run(self, message: PromptValue, logging: bool = False):
        response = self.client.invoke(message)
        
        if logging:
            print(f"Message: {message}")
            print(f"Response: {response}")

        return response
    

class OllamaClient(BaseClient):
    def __init__(
        self,
        model: str,
        context_size: int,
        max_output_tokens: int,
        **krawgs
    ):
        super.__init__(model, context_size, max_output_tokens)

        self.client = OllamaLLM(
            model=model,
            **krawgs
        )
    

    def run(self, message: PromptValue, logging: bool = False):
        response = self.client.invoke(message)
        
        if logging:
            print(f"Message: {message}")
            print(f"Response: {response}")

        return response