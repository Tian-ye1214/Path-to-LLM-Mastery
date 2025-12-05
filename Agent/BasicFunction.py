from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, ModelSettings, ModelProfile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.messages import ModelResponse, ToolCallPart
import json_repair
import os
load_dotenv()


class JsonRepairOpenAIChatModel(OpenAIChatModel):
    async def request(self, *args, **kwargs) -> ModelResponse:
        response = await super().request(*args, **kwargs)
        return self._repair_tool_calls_json(response)
    
    def _repair_tool_calls_json(self, response: ModelResponse) -> ModelResponse:
        """修复响应中所有工具调用的 JSON 参数"""
        repaired_parts = []
        
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                try:
                    original_args = part.args
                    if isinstance(original_args, str):
                        repaired_json = json_repair.loads(original_args)
                        repaired_args = json_repair.dumps(repaired_json)
                    elif isinstance(original_args, dict):
                        repaired_args = json_repair.dumps(original_args)
                    else:
                        repaired_args = original_args
                    part = ToolCallPart(
                        tool_name=part.tool_name,
                        args=repaired_args,
                        tool_call_id=part.tool_call_id,
                        id=part.id,
                        provider_details=part.provider_details,
                    )
                except Exception:
                    pass
            repaired_parts.append(part)

        return ModelResponse(
            parts=repaired_parts,
            usage=response.usage,
            model_name=response.model_name,
            timestamp=response.timestamp,
            provider_name=response.provider_name,
            provider_details=response.provider_details,
            provider_response_id=response.provider_response_id,
            finish_reason=response.finish_reason,
            run_id=response.run_id,
            metadata=response.metadata,
        )


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
    return JsonRepairOpenAIChatModel(
        model_name,
        provider=provider,
        settings=ModelSettings(**parameter)
    )


def create_agent(model_name: str, parameter: dict, tools: list, system_prompt: str):
    if parameter is None:
        parameter = {
            "temperature": 0.6,
            "top_p": 0.8,
            "max_tokens": 8192,
        }

    model = create_model(model_name, parameter)
    agent = Agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent