import pytest

from bodol.providers.base import ToolCall
from bodol.tools import (
    DuplicateToolError,
    InvalidToolError,
    ToolRegistry,
    UnknownToolError,
)


def _weather_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }


def test_register_returns_explicit_model_spec() -> None:
    registry = ToolRegistry()
    spec = registry.register(
        name="weather",
        description="Get the weather.",
        parameters=_weather_schema(),
        handler=lambda city: f"sunny in {city}",
    )

    assert spec.name == "weather"
    assert spec.description == "Get the weather."
    assert registry.specs == (spec,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "", "description": "x"}, "name"),
        ({"name": "x", "description": ""}, "description"),
        ({"name": "x", "description": "x", "parameters": {}}, "Schema"),
    ],
)
def test_register_rejects_invalid_definitions(
    kwargs: dict[str, object],
    message: str,
) -> None:
    registry = ToolRegistry()
    with pytest.raises(InvalidToolError, match=message):
        registry.register(
            parameters=kwargs.pop("parameters", _weather_schema()),  # type: ignore[arg-type]
            handler=lambda: None,
            **kwargs,  # type: ignore[arg-type]
        )


def test_register_rejects_duplicates_and_async_handlers() -> None:
    registry = ToolRegistry()
    registry.register(
        name="weather",
        description="Get weather.",
        parameters=_weather_schema(),
        handler=lambda city: city,
    )

    with pytest.raises(DuplicateToolError):
        registry.register(
            name="weather",
            description="Get weather again.",
            parameters=_weather_schema(),
            handler=lambda city: city,
        )

    async def async_handler() -> None:
        return None

    with pytest.raises(InvalidToolError, match="synchronous"):
        registry.register(
            name="async_tool",
            description="Not supported.",
            parameters={"type": "object"},
            handler=async_handler,
        )


async def test_dispatch_executes_sync_handler() -> None:
    registry = ToolRegistry()
    registry.register(
        name="weather",
        description="Get weather.",
        parameters=_weather_schema(),
        handler=lambda city: {"city": city, "temperature": 72},
    )

    result = await registry.dispatch(ToolCall("call_1", "weather", {"city": "Boston"}))

    assert result.call_id == "call_1"
    # Gemini rejects a function_result that names no tool, so every result
    # block carries the name — not only the ones the model can read an error in.
    assert result.name == "weather"
    assert result.content == '{"city": "Boston", "temperature": 72}'
    assert not result.is_error


async def test_dispatch_invalid_args_returns_model_error() -> None:
    registry = ToolRegistry()
    registry.register(
        name="weather",
        description="Get weather.",
        parameters=_weather_schema(),
        handler=lambda city: city,
    )

    result = await registry.dispatch(ToolCall("call_1", "weather", None, '{"city":'))

    assert result.is_error
    assert result.name == "weather"
    assert '{"city":' in result.content


async def test_dispatch_handler_failure_returns_model_error() -> None:
    def fail() -> None:
        raise RuntimeError("service unavailable")

    registry = ToolRegistry()
    registry.register(
        name="fail",
        description="Fail.",
        parameters={"type": "object"},
        handler=fail,
    )

    result = await registry.dispatch(ToolCall("call_1", "fail", {}))

    assert result.is_error
    assert result.name == "fail"
    assert "service unavailable" in result.content


async def test_dispatch_unknown_tool_raises() -> None:
    with pytest.raises(UnknownToolError):
        await ToolRegistry().dispatch(ToolCall("call_1", "missing", {}))


async def test_dispatch_all_runs_calls_in_parallel() -> None:
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo a value.",
        parameters={"type": "object"},
        handler=lambda value: value,
    )
    calls = [ToolCall(f"call_{i}", "echo", {"value": i}) for i in range(3)]

    results = await registry.dispatch_all(calls)

    assert [result.content for result in results] == ["0", "1", "2"]
