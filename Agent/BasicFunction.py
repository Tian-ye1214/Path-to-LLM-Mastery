from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, ModelSettings, ModelProfile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
import os
load_dotenv()

class CustomProvider(OpenAIProvider):
    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = deepseek_model_profile(model_name)
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            supports_json_object_output=True,
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        ).update(profile)


def create_model(model_name: str, parameter: dict):
    if 'deepseek' in model_name:
        provider = CustomProvider(base_url=os.environ.get('BASE_URL'), api_key=os.environ.get('API_KEY'))
    else:
        provider = OpenAIProvider(
            base_url=os.environ.get('BASE_URL'),
            api_key=os.environ.get('API_KEY')
        )
    return OpenAIChatModel(
        model_name,
        provider=provider,
        settings=ModelSettings(**parameter)
    )


def create_agent(model_name: str, parameter: dict, tools: list, system_prompt: str):
    if parameter is None:
        parameter = {
            "temperature": 0.6,
            "top_p": 0.8,
        }

    model = create_model(model_name, parameter)
    agent = Agent(
        model,
        tools=tools,
        system_prompt=system_prompt
    )
    return agent