"""
Translate from OpenAI's `/v1/chat/completions` to SAP Generative AI Hub's Orchestration Service`v2/completion`
"""
from typing import List, Optional, Union, Dict, Tuple, Any, TYPE_CHECKING, Iterator, AsyncIterator
from functools import cached_property
import litellm
import httpx


from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

from ..credentials import get_token_creator
from .models import (
    SAPMessage,
    SAPAssistantMessage,
    SAPToolChatMessage,
    ChatCompletionTool,
    ResponseFormatJSONSchema,
    ResponseFormat,
    SAPUserMessage,
)
from .handler import GenAIHubOrchestrationError, AsyncSAPStreamIterator, SAPStreamIterator

def validate_dict(data: dict, model) -> dict:
    return model(**data).model_dump(by_alias=True)


class GenAIHubOrchestrationConfig(OpenAIGPTConfig):
    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None  #
    model_version: str = "latest"

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        self.token_creator = None
        self._base_url = None
        self._resource_group = None

    def run_env_setup(self, service_key: Optional[str] = None) -> None:
        try:
            self.token_creator, self._base_url, self._resource_group = get_token_creator(service_key) # type: ignore
        except ValueError as err:
            raise GenAIHubOrchestrationError(status_code=400, message=err.args[0])


    @property
    def headers(self) -> Dict[str, str]:
        if self.token_creator is None:
            self.run_env_setup()
        access_token = self.token_creator() # type: ignore
        return {
            "Authorization": access_token,
            "AI-Resource-Group": self.resource_group,
            "Content-Type": "application/json",
            "AI-Client-Type": "LiteLLM",
        }

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            self.run_env_setup()
        return self._base_url # type: ignore


    @property
    def resource_group(self) -> str:
        if self._resource_group is None:
            self.run_env_setup()
        return self._resource_group # type: ignore

    @cached_property
    def deployment_url(self) -> str:
        # Keep a short, tight client lifecycle here to avoid fd leaks
        client = litellm.module_level_client
        # with httpx.Client(timeout=30) as client:
        deployments = client.get(
            f"{self.base_url}/lm/deployments", headers=self.headers
        ).json()
        valid: List[Tuple[str, str]] = []
        for dep in deployments.get("resources", []):
            if dep.get("scenarioId") == "orchestration":
                cfg = client.get(
                    f'{self.base_url}/lm/configurations/{dep["configurationId"]}',
                    headers=self.headers,
                ).json()
                if cfg.get("executableId") == "orchestration":
                    valid.append((dep["deploymentUrl"], dep["createdAt"]))
            # newest first
        return sorted(valid, key=lambda x: x[1], reverse=True)[0][0]

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model):
        params = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "prediction",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "extra_headers",
            "parallel_tool_calls",
            "response_format",
            "timeout",
        ]
        # Only remove response_format for providers that don't support tool-based JSON mode
        # Note: amazon is kept blocked because SAP removes tool_choice for amazon models,
        # which is required for reliable tool-based JSON mode
        if (
            model.startswith("amazon")
            or model.startswith("cohere")
            or model.startswith("alephalpha")
            or model == "gpt-4"
        ):
            params.remove("response_format")
        if model.startswith("gemini") or model.startswith("amazon"):
            params.remove("tool_choice")
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to SAP-compatible parameters.

        For Anthropic models, converts response_format to tool-based JSON mode
        since SAP Orchestration doesn't pass through response_format natively for these.
        """
        # First, call parent class to handle standard parameter mapping
        mapped_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        # Check if this model needs tool-based JSON mode for response_format
        # Only anthropic models - amazon models don't support tool_choice on SAP
        needs_tool_based_json = model.startswith("anthropic")

        if "response_format" in non_default_params and needs_tool_based_json:
            # Use base class utility to convert response_format to tool call
            # This adds a tool named "json_tool_call" and sets json_mode=True
            mapped_params = self._add_response_format_to_tools(
                optional_params=mapped_params,
                value=non_default_params["response_format"],
                is_response_format_supported=False,  # Force tool conversion
                enforce_tool_choice=True,
            )
            # Remove response_format since it's been converted to tool
            mapped_params.pop("response_format", None)

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if api_key:
            self.run_env_setup(api_key)
        return self.headers

    def get_complete_url(
            self,
            api_base: Optional[str],
            api_key: Optional[str],
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: Optional[bool] = None,
    ):
        api_base_ = f"{self.deployment_url}/v2/completion"
        return api_base_

    def transform_request(
        self,
        model: str,
        messages: List[Dict[str, str]], # type: ignore
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        model_params = {
            k: v for k, v in optional_params.items() if k not in {"tools", "model_version", "deployment_url", "json_mode"}
        }

        model_version = optional_params.pop("model_version", "latest")
        template = []
        for message in messages:
            if message["role"] == "user":
                template.append(validate_dict(message, SAPUserMessage))
            elif message["role"] == "assistant":
                template.append(validate_dict(message, SAPAssistantMessage))
            elif message["role"] == "tool":
                template.append(validate_dict(message, SAPToolChatMessage))
            else:
                template.append(validate_dict(message, SAPMessage))

        tools_ = optional_params.pop("tools", [])
        tools_ = [validate_dict(tool, ChatCompletionTool) for tool in tools_]
        if tools_ != []:
            tools = {"tools": tools_}
        else:
            tools = {}

        response_format = model_params.pop("response_format", {})
        resp_type = response_format.get("type", None)
        if resp_type:
            if resp_type== "json_schema":
                response_format = validate_dict(response_format, ResponseFormatJSONSchema)
            else:
                response_format = validate_dict(response_format, ResponseFormat)
            response_format = {"response_format": response_format}
        model_params.pop("stream", False)
        stream_config = {}
        if "stream_options" in model_params:
            # stream_config["enabled"] = True
            stream_options = model_params.pop("stream_options", {})
            stream_config["chunk_size"] = stream_options.get("chunk_size", 100)
            if "delimiters" in stream_options:
                stream_config["delimiters"] = stream_options.get("delimiters")
        # else:
        #     stream_config["enabled"] = False
        config = {
            "config": {
                "modules": {
                    "prompt_templating": {
                        "prompt": {
                            "template": template,
                            **tools,
                            **response_format
                        },
                        "model": {
                            "name": model,
                            "params": model_params,
                            "version": model_version,
                        },
                    },
                },
                "stream": stream_config,
            }
        }

        return config

    def transform_response(
            self,
            model: str,
            raw_response: httpx.Response,
            model_response: ModelResponse,
            logging_obj: LiteLLMLoggingObj,
            request_data: dict,
            messages: List[AllMessageValues],
            optional_params: dict,
            litellm_params: dict,
            encoding: Any,
            api_key: Optional[str] = None,
            json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )
        response = ModelResponse.model_validate(raw_response.json()["final_result"])

        # Convert json_tool_call response back to message content
        if json_mode or optional_params.get("json_mode"):
            response = self._convert_json_tool_response_to_content(response)

        return response

    def _convert_json_tool_response_to_content(self, response: ModelResponse) -> ModelResponse:
        """
        If the response contains a json_tool_call tool use, convert it to message content.
        This matches OpenAI's response_format behavior.

        Note: Uses dict-style access on tool_calls since LiteLLM types like
        ChatCompletionMessageToolCall and Function support __getitem__.
        """
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _should_convert_tool_call_to_json_mode,
        )

        for choice in response.choices:
            if choice.message and choice.message.tool_calls:
                tool_calls = choice.message.tool_calls
                if _should_convert_tool_call_to_json_mode(
                    tool_calls=tool_calls,
                    convert_tool_call_to_json_mode=True,
                ):
                    # Extract JSON from tool arguments and set as content
                    # tool_calls[0]["function"]["arguments"] uses dict-style access
                    json_content = tool_calls[0]["function"]["arguments"]
                    choice.message.content = json_content
                    choice.message.tool_calls = None
                    choice.finish_reason = "stop"

        return response

    def get_model_response_iterator(
            self,
            streaming_response: Union[Iterator[str], AsyncIterator[str], "ModelResponse"],
            sync_stream: bool,
            json_mode: Optional[bool] = False,
    ):
        if sync_stream:
            return SAPStreamIterator(response=streaming_response, json_mode=json_mode or False)  # type: ignore
        else:
            return AsyncSAPStreamIterator(response=streaming_response, json_mode=json_mode or False)  # type: ignore
