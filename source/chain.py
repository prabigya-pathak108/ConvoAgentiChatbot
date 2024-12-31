from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AZURE_DEPLOYMENT, API_VERSION
from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def get_llm(self, **kwargs):
        NotImplemented
    

class OpenAI(LLM):
    def __init__(self,api_key):
        self.api_key = api_key
    
    def get_llm(self,**kwargs):
        """
        Get the LLM for the current session.
        Args:
            llm (ChatOpenAI): The OpenAI model to use.
        Returns:
            LLMChain: A LLMChain object.
        """
        temperature = kwargs.get("temperature") if kwargs.get("temperature") else ValueError("Temperature is required")
        model = kwargs.get("model") if kwargs.get("model") else ValueError("Model is required")
        streaming = kwargs.get("streaming") if kwargs.get("streaming") else False
        return ChatOpenAI(api_key=self.api_key,temperature=temperature, model=model, streaming=streaming)

class AzureOpenAI(LLM):
    def __init__(self,api_key):
        self.api_key = api_key

    def get_llm(self,**kwargs):
        """
        Get the LLM for the current session.
        Args:
            llm (ChatOpenAI): The OpenAI model to use.
        Returns:
            LLMChain: A LLMChain object.
        """
        temperature = kwargs.get("temperature") if kwargs.get("temperature") else ValueError("Temperature is required")
        model = kwargs.get("model") if kwargs.get("model") else ValueError("Model is required")
        return AzureChatOpenAI(api_key=self.api_key,
            azure_deployment=AZURE_DEPLOYMENT,
            api_version=API_VERSION,
            temperature=temperature,
            model=model
        )

class GeminiOpenAI(LLM):
    def __init__(self,api_key):
        self.api_key = api_key

    def get_llm(self,**kwargs):
        """
        Get the LLM for the current session.
        Args:
            llm (ChatOpenAI): The OpenAI model to use.
        Returns:
            LLMChain: A LLMChain object.
        """
        temperature = kwargs.get("temperature") if kwargs.get("temperature") else ValueError("Temperature is required")
        model = kwargs.get("model") if kwargs.get("model") else ValueError("Model is required")
        streaming = kwargs.get("streaming") if kwargs.get("streaming") else False
        return ChatGoogleGenerativeAI(api_key=self.api_key,temperature=temperature, model=model, streaming=streaming)


class LLMFactory():
    def __init__(self, type,api_key):
        self.type = type
        self.llms = {
            "azure": AzureOpenAI(api_key=api_key),
            "openai": OpenAI(api_key=api_key),
            "gemini": GeminiOpenAI(api_key=api_key)
        }

    def get_llm(self, temperature, model, streaming):
        return self.llms[self.type].get_llm(temperature=temperature, model=model, streaming=streaming)


def get_chain(LLM_TYPE,api_key,temperature, model, prompt):
    """
    Get the conversational chain for the current session.
    Args:
        llm (ChatOpenAI): The OpenAI model to use.
        memory (ConversationBufferWindowMemory): The memory object for the current session.
        prompt (str): The prompt template to use.
    Returns:
        LLMChain: A LLMChain object.
    """
    prompt = PromptTemplate(input_variables=["input", "history"], template=prompt)
    llm = LLMFactory(LLM_TYPE,api_key).get_llm(temperature=temperature, model=model, streaming=True)
    chain = prompt | llm
    return chain
