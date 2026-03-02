"""
Test SAP response_format support for Anthropic models.

This test ensures that response_format is properly converted to tool-based JSON mode
for Anthropic models on SAP GenAI Hub, following the same pattern as Databricks.
"""

import pytest
from unittest.mock import MagicMock, patch

from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.types.utils import ModelResponse, Choices, Message
import httpx


class TestResponseFormatSupport:
    """Test response_format is supported for Anthropic models."""

    def test_anthropic_model_supports_response_format(self):
        """Anthropic models should support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("anthropic--claude-3-5-sonnet")
        assert "response_format" in params

    def test_cohere_model_does_not_support_response_format(self):
        """Cohere models should not support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("cohere--command-r")
        assert "response_format" not in params

    def test_amazon_model_does_not_support_response_format(self):
        """Amazon models should not support response_format (no tool_choice on SAP)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("amazon--nova-pro")
        assert "response_format" not in params

    def test_alephalpha_model_does_not_support_response_format(self):
        """AlephAlpha models should not support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("alephalpha--luminous")
        assert "response_format" not in params

    def test_gpt4_exact_does_not_support_response_format(self):
        """gpt-4 (exact match) should not support response_format param."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("gpt-4")
        assert "response_format" not in params

    def test_gpt4o_supports_response_format(self):
        """gpt-4o should support response_format (native support)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("gpt-4o")
        assert "response_format" in params

    def test_gemini_supports_response_format(self):
        """Gemini models should support response_format (native support)."""
        config = GenAIHubOrchestrationConfig()
        params = config.get_supported_openai_params("gemini-1.5-pro")
        assert "response_format" in params


class TestMapOpenAIParamsJsonMode:
    """Test map_openai_params creates tool-based JSON mode for Anthropic models."""

    def test_map_openai_params_creates_json_tool_for_anthropic(self):
        """Anthropic models should convert response_format to tool call."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}}
                }
            }
        }
        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        assert "tools" in optional_params
        assert optional_params.get("json_mode") is True
        assert any(
            t["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME
            for t in optional_params["tools"]
        )
        # response_format should be removed after conversion
        assert "response_format" not in optional_params

    def test_map_openai_params_sets_tool_choice_for_anthropic(self):
        """Anthropic models should have tool_choice set to force the JSON tool."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}}
                }
            }
        }
        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        assert "tool_choice" in optional_params
        assert optional_params["tool_choice"]["type"] == "function"
        assert optional_params["tool_choice"]["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME

    def test_map_openai_params_preserves_response_format_for_gpt4o(self):
        """GPT-4o should keep response_format (native support)."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}}
            }
        }
        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="gpt-4o",
            drop_params=False,
        )

        # response_format should be preserved, not converted to tools
        assert "response_format" in optional_params
        assert optional_params.get("json_mode") is not True

    def test_map_openai_params_handles_response_schema_format(self):
        """Should handle response_schema format (alternative to json_schema)."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "response_schema": {
                "type": "object",
                "properties": {"data": {"type": "string"}}
            }
        }
        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        assert "tools" in optional_params
        assert optional_params.get("json_mode") is True


class TestResponseTransformation:
    """Test response transformation for JSON mode."""

    def test_converts_tool_response_to_content(self):
        """Tool response with json_tool_call should be converted to message content."""
        config = GenAIHubOrchestrationConfig()

        # Create a mock tool call response
        tool_call = MagicMock()
        tool_call.__getitem__ = lambda self, key: {
            "function": {"name": RESPONSE_FORMAT_TOOL_NAME, "arguments": '{"result": "test"}'}
        }[key]

        message = MagicMock(spec=Message)
        message.tool_calls = [tool_call]
        message.content = None

        choice = MagicMock(spec=Choices)
        choice.message = message
        choice.finish_reason = "tool_calls"

        response = MagicMock(spec=ModelResponse)
        response.choices = [choice]

        result = config._convert_json_tool_response_to_content(response)

        # Should have converted to content
        assert result.choices[0].message.content == '{"result": "test"}'
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"

    def test_does_not_convert_regular_tool_calls(self):
        """Regular tool calls (not json_tool_call) should be preserved."""
        config = GenAIHubOrchestrationConfig()

        # Create a mock tool call response with different name
        tool_call = MagicMock()
        tool_call.__getitem__ = lambda self, key: {
            "function": {"name": "web_search", "arguments": '{"query": "test"}'}
        }[key]

        message = MagicMock(spec=Message)
        message.tool_calls = [tool_call]
        message.content = None

        choice = MagicMock(spec=Choices)
        choice.message = message
        choice.finish_reason = "tool_calls"

        response = MagicMock(spec=ModelResponse)
        response.choices = [choice]

        result = config._convert_json_tool_response_to_content(response)

        # Should NOT have converted - tool_calls should still exist
        assert result.choices[0].message.tool_calls is not None


class TestStreamingJsonMode:
    """Test streaming support for JSON mode."""

    def test_sync_stream_iterator_accepts_json_mode(self):
        """SAPStreamIterator should accept json_mode parameter."""
        from litellm.llms.sap.chat.handler import SAPStreamIterator

        iterator = SAPStreamIterator(response=iter([]), json_mode=True)
        assert iterator.json_mode is True

    def test_async_stream_iterator_accepts_json_mode(self):
        """AsyncSAPStreamIterator should accept json_mode parameter."""
        from litellm.llms.sap.chat.handler import AsyncSAPStreamIterator

        async def async_gen():
            yield ""

        iterator = AsyncSAPStreamIterator(response=async_gen(), json_mode=True)
        assert iterator.json_mode is True

    def test_get_model_response_iterator_passes_json_mode(self):
        """get_model_response_iterator should pass json_mode to iterators."""
        config = GenAIHubOrchestrationConfig()

        iterator = config.get_model_response_iterator(
            streaming_response=iter([]),
            sync_stream=True,
            json_mode=True,
        )

        assert iterator.json_mode is True


class TestTransformRequestWithJsonMode:
    """Test transform_request handles json_mode tools correctly."""

    def test_transform_request_includes_json_tool_for_anthropic(self):
        """transform_request should include the json_tool_call tool for anthropic models."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}}
                }
            }
        }

        # First, map params (this adds the tool)
        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        # Then transform request
        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # Verify tools are in the request
        prompt_config = request["config"]["modules"]["prompt_templating"]["prompt"]
        assert "tools" in prompt_config
        assert len(prompt_config["tools"]) == 1
        assert prompt_config["tools"][0]["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME

        # Verify response_format is NOT in the request (it was converted to tool)
        assert "response_format" not in prompt_config

    def test_transform_request_excludes_json_mode_from_model_params(self):
        """transform_request should not include json_mode in model params."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}}
            }
        }

        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        request = config.transform_request(
            model="anthropic--claude-3-5-sonnet",
            messages=[{"role": "user", "content": "Test"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        # json_mode should not appear in the model params
        model_params = request["config"]["modules"]["prompt_templating"]["model"]["params"]
        assert "json_mode" not in model_params


class TestEndToEndJsonModeWithMockedAPI:
    """End-to-end tests with mocked SAP API responses."""

    @pytest.fixture
    def mock_anthropic_tool_response(self):
        """Mock SAP API response with tool call (simulating json_tool_call)."""
        return {
            "request_id": "test-request-id",
            "intermediate_results": {},
            "final_result": {
                "id": "msg_test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "anthropic--claude-3-5-sonnet",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "toolu_test",
                            "type": "function",
                            "function": {
                                "name": RESPONSE_FORMAT_TOOL_NAME,
                                "arguments": '{"result": "test_value", "count": 42}'
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30}
            }
        }

    @pytest.fixture
    def mock_anthropic_regular_tool_response(self):
        """Mock SAP API response with regular tool call (not json_tool_call)."""
        return {
            "request_id": "test-request-id",
            "intermediate_results": {},
            "final_result": {
                "id": "msg_test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "anthropic--claude-3-5-sonnet",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "toolu_test",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}'
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30}
            }
        }

    def test_transform_response_converts_json_tool_to_content(self, mock_anthropic_tool_response):
        """Full flow: response with json_tool_call should have content extracted."""
        import httpx
        config = GenAIHubOrchestrationConfig()

        # Create mock httpx response
        raw_response = httpx.Response(
            200,
            json=mock_anthropic_tool_response,
            request=httpx.Request("POST", "https://mock.sap.com/v2/completion")
        )

        # Transform response with json_mode=True
        result = config.transform_response(
            model="anthropic--claude-3-5-sonnet",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={"json_mode": True},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=True,
        )

        # Verify the tool call was converted to content
        assert result.choices[0].message.content == '{"result": "test_value", "count": 42}'
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"

    def test_transform_response_preserves_regular_tool_calls(self, mock_anthropic_regular_tool_response):
        """Regular tool calls (not json_tool_call) should be preserved."""
        import httpx
        config = GenAIHubOrchestrationConfig()

        raw_response = httpx.Response(
            200,
            json=mock_anthropic_regular_tool_response,
            request=httpx.Request("POST", "https://mock.sap.com/v2/completion")
        )

        # Transform response with json_mode=False (or without it)
        result = config.transform_response(
            model="anthropic--claude-3-5-sonnet",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
            api_key=None,
            json_mode=False,
        )

        # Verify the tool call is preserved
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"

    def test_json_tool_schema_has_type_object(self):
        """The generated json_tool_call should have type='object' in parameters."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}}
                }
            }
        }

        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        # Find the json_tool_call tool
        json_tool = next(
            t for t in optional_params["tools"]
            if t["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME
        )

        # Verify it has type='object' (required by SAP)
        assert json_tool["function"]["parameters"]["type"] == "object"

    def test_json_mode_with_existing_user_tools(self):
        """When user provides tools AND response_format, both should be included."""
        config = GenAIHubOrchestrationConfig()

        user_tools = [{
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }]

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}}
                }
            }
        }

        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format, "tools": user_tools},
            optional_params={"tools": user_tools},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        # Should have both user tool and json_tool_call
        tool_names = [t["function"]["name"] for t in optional_params["tools"]]
        assert "search_web" in tool_names
        assert RESPONSE_FORMAT_TOOL_NAME in tool_names
        assert len(optional_params["tools"]) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_response_format_type_json_object(self):
        """Test response_format with type='json_object' (simpler format)."""
        config = GenAIHubOrchestrationConfig()
        response_format = {"type": "json_object"}

        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        # json_object type without schema should not create tools
        # (no schema to convert)
        assert optional_params.get("json_mode") is not True

    def test_empty_schema(self):
        """Test with empty schema object - should not create tool."""
        config = GenAIHubOrchestrationConfig()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "empty",
                "schema": {}
            }
        }

        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        # Empty schema should not create tools (base class behavior)
        # This is correct - an empty schema provides no structure
        assert "tools" not in optional_params or optional_params.get("json_mode") is not True

    def test_nested_schema_preserved(self):
        """Test that complex nested schemas are preserved correctly."""
        config = GenAIHubOrchestrationConfig()
        nested_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "addresses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "nested",
                "schema": nested_schema
            }
        }

        optional_params = config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={},
            model="anthropic--claude-3-5-sonnet",
            drop_params=False,
        )

        # Verify the nested schema is preserved
        json_tool = next(
            t for t in optional_params["tools"]
            if t["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME
        )
        assert json_tool["function"]["parameters"] == nested_schema
