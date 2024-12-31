from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from dateutil import parser
from datetime import datetime, timedelta
import spacy
from langchain.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field
from typing import List, Type, Union, Optional
from langchain.agents import AgentExecutor, create_tool_calling_agent
import random

nlp = spacy.load("en_core_web_sm")

class NameField(BaseModel):
    text: str = Field(description="should be a search query")
class EmailField(BaseModel):
    email: str = Field(description="email of user in search query")
class DateField(BaseModel):
    date_string: str = Field(description="absolute or relative date field in search query")

class EmailValidationTool(BaseTool):
    name: str = "email_extractor"
    description: str = "Extract email from text"
    args_schema: Type[BaseModel] =EmailField
    return_direct: bool = True
    def _run(self, email: str) -> str:
        pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if re.match(pattern, email):
            return f"{email}"
        return "Invalid Entry: Invalid email format"
    async def _arun(self, email: str) -> str:
       raise NotImplementedError("This tool does not support async")

class DateValidationTool(BaseTool):
    name:str = "date_extractor"
    description:str = "Extract date from user query, retrieve date part both relative and absolute date part like today, tomorrow, next Sunday, Jan 2, etc. Pass in standard numerical format to tool."
    args_schema: Type[BaseModel] = DateField
    return_direct: bool = True

    def _run(self, date_string: str) -> str:
        try:
            relative_dates = {
                'today': datetime.now(),
                'tomorrow': datetime.now() + timedelta(days=1),
                'next week': datetime.now() + timedelta(days=7),
                "day after tomorrow": datetime.now() + timedelta(days=2),
                'next monday': datetime.now() + timedelta(days=(7 - datetime.now().weekday() + 0) % 7),
                'next tuesday': datetime.now() + timedelta(days=(7 - datetime.now().weekday() + 1) % 7),
                'next wednesday': datetime.now() + timedelta(days=(7 - datetime.now().weekday() + 2) % 7),
                'next thursday': datetime.now() + timedelta(days=(7 - datetime.now().weekday() + 3) % 7),
                'next friday': datetime.now() + timedelta(days=(7 - datetime.now().weekday() + 4) % 7),
                'next saturday': datetime.now() + timedelta(days=(7 - datetime.now().weekday() + 5) % 7),
                'next sunday': datetime.now() + timedelta(days=(7 - datetime.now().weekday() + 6) % 7),
            }
            
            if date_string.lower() in relative_dates:
                return relative_dates[date_string.lower()].strftime('%Y-%m-%d')
                
            parsed_date = parser.parse(date_string)
            if parsed_date< datetime.now():
                return "Invalid Entry: Could not parse the date. Please use a clear date format."
            return parsed_date.strftime('%Y-%m-%d')
        except Exception:
            return "Invalid Entry: Could not parse the date. Please use a clear date format."

    def _arun(self, date_string: str):
        raise NotImplementedError("This tool does not support async")

class NameExtractionTool(BaseTool):
    name:str = "name_extractor"
    description:str = "Extract name of person from text"
    args_schema: Type[BaseModel] =NameField
    return_direct: bool = True

    def extract_name(self,text):
        match = re.search(r"\b(?:my name is|this is|i am|i'm|i'm called|under the name)\s+([a-zA-Z]+(?:\s[a-zA-Z]+)*)",text, re.IGNORECASE)
        return match.group(1) if match else None

    def extract_name_with_spacy(self,text):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None
    
    def _run(self, text: str) -> str:
        nlp_name=self.extract_name_with_spacy(text)
        if nlp_name:
            return nlp_name
        simple_name=self.extract_name(text)
        if simple_name:
            return simple_name
        return text

class UserInfoCollectorWithToolAndAgent:
    def __init__(self, api_key,model):
        self.api_key = api_key
        self.model_name=model
        self.user_info = {"name": {"value":None,"Error":False}, "email": {"value":None,"Error":False}, "date": {"value":None,"Error":False}}
        self.memory=ConversationBufferMemory(memory_key="chat_history")
        self.tools = self._initialize_tools()
        self.tool_mapping=self._initialize_tool_mapper()
        self.agent = self._initialize_agent()
        
        
    def _initialize_tool_mapper(self):
        tool_mapping={}
        for tool in self.tools:
            tool_mapping[tool.name]=tool
        return tool_mapping

    def _initialize_tools(self) -> List[BaseTool]:
        return [
            EmailValidationTool(),
            DateValidationTool(),
            NameExtractionTool()
        ]
    
    def _initialize_agent(self):
        llm = ChatGoogleGenerativeAI(api_key=self.api_key, model=self.model_name,temperature=0.1)
        llm_with_tools=llm.bind_tools(self.tools)
        return llm_with_tools

    def _update_user_info(self, tool_name, result):
        """Update the user_info dictionary based on tool output."""
        if tool_name == "name_extractor" and result:
            self.user_info["name"] = result
        elif tool_name == "email_validator" and result != "Invalid email format":
            self.user_info["email"] = result
        elif tool_name == "date_validator" and result != "Invalid date":
            self.user_info["date"] = result
    
    def _get_question_for_field(self, field, error=False):
        questions = {
            "name": {
                "error": ["The name provided is invalid. Can you please provide a valid name?"],
                "none": ["Please tell me your name:", "Under what name can I book the appointment?"]
            },
            "email": {
                "error": ["The email address is incorrect. Please provide a valid email address."],
                "none": ["Please provide your email address:"]
            },
            "date": {
                "error": ["The date provided is not valid. Could you specify a correct date for the appointment?"],
                "none": ["Specify a date for appointment.", "When can I schedule the appointment?"]
            }
        }

        if error:
            return questions.get(field, {}).get("error", ["Invalid input for this field."])
        else:
            return questions.get(field, {}).get("none", ["No value provided. Please fill in this field."])
    
    def process_user_input(self, user_input):
        result=self.agent.invoke(user_input)
        for tool_call in result.tool_calls:
            tool = self.tool_mapping[tool_call["name"].lower()]
            tool_output = tool.invoke(tool_call["args"])
            if "Invalid Entry:" in tool_output:
                self.user_info[str(tool.name).replace("_extractor", "")]["Error"] = True

            else:
                self.user_info[str(tool.name).replace("_extractor","")]["value"]= tool_output
                self.user_info[str(tool.name).replace("_extractor", "")]["Error"] = False
    def final_appointment_text(self):
        text = f"""Your appointment is booked with the following information:
                Name: {self.user_info["name"]["value"]}
                Email: {self.user_info["email"]["value"]}
                Date: {self.user_info["date"]["value"]}\n"""
        return text
    def collect_user_information(self,initial_query):
        # self.process_user_input(initial_query)
        # print("-------------------")
        # print(self.user_info)
        # print("-------------------")
        def find_incomplete_field():
            for field, data in self.user_info.items():
                if data["Error"]:
                    return field, True 
                elif data["value"] is None:
                    return field, False  
            return None, None

        while True:
            field, has_error = find_incomplete_field()
            if field is None:
                break  
            
            if has_error:
                error_questions = self._get_question_for_field(field, error=True)
                question = random.choice(error_questions)
                print(f"\x1b[34mAI: \x1b[0m: {question}")
            else:
                none_questions = self._get_question_for_field(field, error=False)
                question = random.choice(none_questions)
                print(f"\x1b[34mAI: \x1b[0m: {question}")
            
            # Assume user provides input and we handle it (pseudo-code)
            print("\x1b[34mHuman Response: \x1b[0m ",end="")
            user_input = input()
            print(user_input)

            self.process_user_input(user_input)
