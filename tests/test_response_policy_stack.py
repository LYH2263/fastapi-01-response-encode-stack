"""Acceptance tests for the request scoped response serialization policy stack.

The tests are split by exit point:

1. stack primitives (push/pop/context manager/fold/restore)
2. plain JSON (``serialize_response``)
3. JSONL streams (per item)
4. SSE streams (per item)
5. WebSocket (text frames, validation failure -> close reason)
6. default exception handlers (must not subscribe to the stack)
7. OpenAPI / JSON Schema (decorator fields only)
8. BackgroundTasks (empty stack view after the response is sent)
9. concurrency isolation (ASGI transport, streaming interleaving, TestClient)
10. dependency_overrides coexistence
11. middleware inherited frames
12. old, never-pushed routes keep behaving exactly like before
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from typing import Annotated, Any

import anyio
import pytest
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    ResponseSerializationPolicy,
    ValidationMode,
    WebSocket,
    get_pushed_response_policy,
    pop_response_policy,
    push_response_policy,
    response_policy,
)
from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, ConfigDict, Field
from starlette.websockets import WebSocketDisconnect


class Item(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(alias="NAMED")
    nickname: str | None = None
    price: float = 10.0


# ---------------------------------------------------------------------------
# 1. Stack primitives
# ---------------------------------------------------------------------------


def test_empty_stack_has_no_pushed_policy():
    assert get_pushed_response_policy() is None


def test_push_folds_and_pop_restores_outer_user_frame():
    t1 = push_response_policy(exclude_none=True)
    try:
        assert get_pushed_response_policy() == ResponseSerializationPolicy(
            exclude_none=True
        )
        t2 = push_response_policy(by_alias=False)
        try:
            folded = get_pushed_response_policy()
            assert folded is not None
            assert folded.exclude_none is True
            assert folded.by_alias is False
        finally:
            pop_response_policy(t2)
        # Inner pop restores the OUTTER user frame, not the decorator defaults:
        # exclude_none stays on, by_alias returns to "unset".
        folded = get_pushed_response_policy()
        assert folded == ResponseSerializationPolicy(exclude_none=True)
        assert folded.by_alias is None
    finally:
        pop_response_policy(t1)
    assert get_pushed_response_policy() is None


def test_context_manager_pops_on_exception():
    with pytest.raises(RuntimeError, match="boom"):
        with response_policy(exclude_none=True):
            assert get_pushed_response_policy() is not None
            raise RuntimeError("boom")
    assert get_pushed_response_policy() is None

    # The frame is really gone: a later push sees an empty stack.
    token = push_response_policy(by_alias=False)
    try:
        folded = get_pushed_response_policy()
        assert folded is not None
        assert folded.exclude_none is None
        assert folded.by_alias is False
    finally:
        pop_response_policy(token)


def test_context_manager_can_be_nested_and_restores_outer_frame():
    with response_policy(exclude_none=True):
        with response_policy(validation_mode=ValidationMode.lenient):
            folded = get_pushed_response_policy()
            assert folded is not None
            assert folded.exclude_none is True
            assert folded.validation_mode is ValidationMode.lenient
        folded = get_pushed_response_policy()
        assert folded == ResponseSerializationPolicy(
            exclude_none=True,
            validation_mode=None,
        )
    assert get_pushed_response_policy() is None


def test_pop_on_empty_stack_and_mismatched_pop_raise():
    token = push_response_policy(exclude_none=True)
    other = push_response_policy(by_alias=False)
    pop_response_policy(other)
    with pytest.raises(RuntimeError, match="pushed on top"):
        pop_response_policy(other)
    pop_response_policy(token)
    with pytest.raises(RuntimeError, match="stack is empty"):
        pop_response_policy(token)


def test_policy_frame_and_string_validation_mode():
    frame = ResponseSerializationPolicy(exclude_none=True, validation_mode="lenient")
    with response_policy(frame):
        folded = get_pushed_response_policy()
        assert folded is not None
        assert folded.exclude_none is True
        assert folded.validation_mode is ValidationMode.lenient
    with pytest.raises(ValueError, match="validation_mode"):
        push_response_policy(validation_mode="weird")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        push_response_policy(unknown_switch=True)


# ---------------------------------------------------------------------------
# 2. Plain JSON exit: decorator baseline merged with the current stack top
# ---------------------------------------------------------------------------


def test_plain_json_without_push_is_unchanged():
    app = FastAPI()

    @app.get("/item", response_model=Item)
    async def get_item() -> Any:
        return Item(name="x", nickname=None)

    client = TestClient(app)
    assert client.get("/item").json() == {"NAMED": "x", "nickname": None, "price": 10.0}


def test_dependency_push_tightens_exclude_none_for_response():
    app = FastAPI()

    def drop_none() -> None:
        # A raw push survives until the framework serializes the response;
        # the request bind discards the frame when the request ends.
        push_response_policy(exclude_none=True)

    @app.get("/item", response_model=Item, dependencies=[Depends(drop_none)])
    async def get_item() -> Any:
        return Item(name="x", nickname=None)

    @app.get("/raw", dependencies=[Depends(drop_none)])
    async def get_raw():
        # No return annotation: no response field, so the jsonable_encoder
        # exit branch is used.
        return {"name": "x", "nickname": None}

    client = TestClient(app)
    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}
    # Also applies to endpoints without a response model.
    assert client.get("/raw").json() == {"name": "x"}
    # Request bind cleanup: nothing leaks outside the request.
    assert get_pushed_response_policy() is None


def test_sync_endpoint_push_in_worker_thread_reaches_serialization():
    app = FastAPI()

    def drop_none() -> None:
        push_response_policy(exclude_none=True)

    @app.get("/item", response_model=Item, dependencies=[Depends(drop_none)])
    def get_item() -> Any:
        return Item(name="x", nickname=None)

    @app.get("/item2", response_model=Item)
    def get_item2() -> Any:
        push_response_policy(exclude_none=True)
        return Item(name="y", nickname=None)

    client = TestClient(app)
    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}
    assert client.get("/item2").json() == {"NAMED": "y", "price": 10.0}


def test_stack_overrides_decorator_baseline_and_merges_with_it():
    app = FastAPI()

    # Decorator baseline: alias on, exclude_none on.
    @app.get(
        "/item",
        response_model=Item,
        response_model_exclude_none=True,
    )
    async def get_item() -> Any:
        return Item(name="x", nickname=None)

    # Same baseline but a frame relaxes by_alias only.
    @app.get(
        "/item-flat",
        response_model=Item,
        response_model_exclude_none=True,
        dependencies=[
            Depends(lambda: push_response_policy(by_alias=False)),
        ],
    )
    async def get_item_flat() -> Any:
        return Item(name="x", nickname=None)

    # Frame explicitly re-enables None output, overriding the decorator.
    @app.get(
        "/item-keep-none",
        response_model=Item,
        response_model_exclude_none=True,
        dependencies=[
            Depends(lambda: push_response_policy(exclude_none=False)),
        ],
    )
    async def get_item_keep_none() -> Any:
        return Item(name="x", nickname=None)

    client = TestClient(app)
    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}
    assert client.get("/item-flat").json() == {"name": "x", "price": 10.0}
    assert client.get("/item-keep-none").json() == {
        "NAMED": "x",
        "nickname": None,
        "price": 10.0,
    }


def test_decorator_exclude_unset_combined_with_stack_exclude_none():
    app = FastAPI()

    def drop_none() -> None:
        push_response_policy(exclude_none=True)

    @app.post(
        "/item",
        response_model=Item,
        response_model_exclude_unset=True,
        dependencies=[Depends(drop_none)],
    )
    async def make_item() -> Any:
        return Item.model_validate({"NAMED": "x"})

    client = TestClient(app)
    # nickname is unset AND none; price is a default -> excluded by
    # exclude_unset; the frame does not undo the decorator switch.
    assert client.post("/item").json() == {"NAMED": "x"}


def test_nested_push_pop_inside_endpoint_returns_to_outer_frame():
    app = FastAPI()

    def outer() -> None:
        push_response_policy(exclude_none=True)

    @app.get("/item", response_model=Item, dependencies=[Depends(outer)])
    async def get_item(request: Request) -> Any:
        # Tighter section inside the endpoint: it must not survive.
        with response_policy(by_alias=False):
            folded = get_pushed_response_policy()
            assert folded is not None
            assert folded.exclude_none is True
            assert folded.by_alias is False
        folded = get_pushed_response_policy()
        assert folded is not None
        assert folded.exclude_none is True
        assert folded.by_alias is None
        return Item(name="x", nickname=None)

    client = TestClient(app)
    # Outer frame still governs the returned response: aliases on, none off.
    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}


def test_strict_mode_still_raises_response_validation_error():
    app = FastAPI()

    @app.get("/item", response_model=Item)
    async def get_item() -> Any:
        return {"name": 123, "nickname": None}

    client = TestClient(app)
    with pytest.raises(ResponseValidationError):
        client.get("/item")


def test_lenient_mode_skips_response_validation_but_still_serializes():
    app = FastAPI()

    @app.get(
        "/item",
        response_model=Item,
        dependencies=[
            Depends(
                lambda: push_response_policy(
                    exclude_none=True,
                    validation_mode=ValidationMode.lenient,
                )
            )
        ],
    )
    async def get_item() -> Any:
        # Wrong type for `name`; lenient mode must not validate, but the
        # exclude_none switch still applies at serialization.
        return {"name": 123, "nickname": None}

    client = TestClient(app)
    response = client.get("/item")
    assert response.status_code == 200
    body = response.json()
    assert body["NAMED"] == 123
    assert "nickname" not in body


def test_lenient_frame_popped_reverts_to_strict_for_later_section():
    app = FastAPI()

    @app.get(
        "/item",
        response_model=Item,
        dependencies=[
            Depends(lambda: push_response_policy(exclude_none=True)),
        ],
    )
    async def get_item() -> Any:
        with response_policy(validation_mode=ValidationMode.lenient):
            assert (
                get_pushed_response_policy().validation_mode is ValidationMode.lenient
            )
        folded = get_pushed_response_policy()
        assert folded is not None
        assert folded.validation_mode is None
        return Item(name="x", nickname=None)

    client = TestClient(app)
    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}


# ---------------------------------------------------------------------------
# 3. JSONL streams: stack top is read per item, never snapshotted at start
# ---------------------------------------------------------------------------


def _data_lines(text: str, prefix: str = "data: ") -> list[str]:
    return [line.removeprefix(prefix) for line in text.splitlines() if line]


def test_jsonl_reads_stack_top_per_item_not_startup_snapshot():
    app = FastAPI()

    @app.get("/items")
    async def stream_items():
        # Untyped generator (jsonable_encoder per-item path). Item 1: stack
        # empty at generator startup -> null kept.
        yield {"name": "a", "nickname": None}
        # Frame pushed AFTER the generator has started iterating.
        token = push_response_policy(exclude_none=True)
        yield {"name": "b", "nickname": None}
        pop_response_policy(token)
        # Frame popped -> null kept again.
        yield {"name": "c", "nickname": None}

    client = TestClient(app)
    response = client.get("/items")
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert lines == [
        {"name": "a", "nickname": None},
        {"name": "b"},
        {"name": "c", "nickname": None},
    ]


def test_jsonl_context_manager_frame_is_active_while_item_is_written():
    app = FastAPI()

    @app.get("/items")
    async def stream_items():
        with response_policy(exclude_none=True):
            # Serialization happens while the generator is suspended at the
            # yield, i.e. while the frame is active.
            yield {"name": "a", "nickname": None}
        # Generator resumed -> frame popped.
        yield {"name": "b", "nickname": None}

    client = TestClient(app)
    lines = [
        json.loads(line) for line in client.get("/items").text.splitlines() if line
    ]
    assert lines == [{"name": "a"}, {"name": "b", "nickname": None}]


def test_jsonl_sync_generator_frame_in_worker_thread_applies_per_item():
    app = FastAPI()

    @app.get("/items")
    def stream_items():
        with response_policy(exclude_none=True):
            yield {"name": "a", "nickname": None}
        yield {"name": "b", "nickname": None}

    client = TestClient(app)
    lines = [
        json.loads(line) for line in client.get("/items").text.splitlines() if line
    ]
    assert lines == [{"name": "a"}, {"name": "b", "nickname": None}]


def test_jsonl_typed_stream_strict_rejects_bad_item_after_lenient_pop():
    app = FastAPI()

    @app.get("/items-ok")
    async def stream_ok() -> AsyncIterator[Item]:
        push_response_policy(validation_mode=ValidationMode.lenient)
        yield {"name": "a", "price": "not-a-number"}
        yield {"name": "b", "price": "also-bad"}

    @app.get("/items-mixed")
    async def stream_mixed() -> AsyncIterator[Item]:
        yield {"name": "a", "price": 1.0}
        with response_policy(validation_mode=ValidationMode.lenient):
            yield {"name": "b", "price": "not-a-number"}
        # Frame popped -> strict baseline returns; bad item must raise.
        yield {"name": "c", "price": "also-bad"}

    client = TestClient(app)
    response = client.get("/items-ok")
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line]
    assert [line["NAMED"] for line in lines] == ["a", "b"]

    with pytest.raises(ResponseValidationError):
        client.get("/items-mixed")


def test_jsonl_typed_stream_item_switches_apply_per_item():
    app = FastAPI()

    @app.get("/items")
    async def stream_items() -> AsyncIterator[Item]:
        with response_policy(exclude_none=True):
            yield Item(name="a", nickname=None)
        yield Item(name="b", nickname=None)

    client = TestClient(app)
    lines = [
        json.loads(line) for line in client.get("/items").text.splitlines() if line
    ]
    assert lines == [
        {"NAMED": "a", "price": 10.0},
        {"NAMED": "b", "nickname": None, "price": 10.0},
    ]


# ---------------------------------------------------------------------------
# 4. SSE streams: same per-item semantics, SSE wire format preserved
# ---------------------------------------------------------------------------


def test_sse_plain_items_read_stack_top_per_item():
    from fastapi.sse import EventSourceResponse

    app = FastAPI()

    @app.get("/events", response_class=EventSourceResponse)
    async def events():
        yield {"name": "a", "nickname": None}
        with response_policy(exclude_none=True):
            yield {"name": "b", "nickname": None}
        yield {"name": "c", "nickname": None}

    client = TestClient(app)
    response = client.get("/events")
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    payloads = [json.loads(line) for line in _data_lines(response.text)]
    assert payloads == [
        {"name": "a", "nickname": None},
        {"name": "b"},
        {"name": "c", "nickname": None},
    ]


def test_sse_server_sent_event_model_data_uses_stack_switches():
    from fastapi.sse import EventSourceResponse, ServerSentEvent

    app = FastAPI()

    @app.get("/events", response_class=EventSourceResponse)
    async def events():
        with response_policy(exclude_none=True):
            yield ServerSentEvent(data=Item(name="a", nickname=None))
        yield ServerSentEvent(data=Item(name="b", nickname=None))

    client = TestClient(app)
    lines = _data_lines(client.get("/events").text)
    # Frame active: exclude_none + baseline by_alias on.
    assert json.loads(lines[0]) == {"NAMED": "a", "price": 10.0}
    # Frame popped: plain model_dump_json() historical behavior keeps the
    # null and uses field names (pydantic's model_dump_json default).
    assert json.loads(lines[1]) == {"name": "b", "nickname": None, "price": 10.0}


def test_sse_raw_data_is_touched_by_no_policy():
    from fastapi.sse import EventSourceResponse, ServerSentEvent

    app = FastAPI()

    @app.get("/events", response_class=EventSourceResponse)
    async def events():
        with response_policy(exclude_none=True):
            yield ServerSentEvent(raw_data="plain text, no json")

    client = TestClient(app)
    assert "data: plain text, no json\n" in client.get("/events").text


# ---------------------------------------------------------------------------
# 5. WebSocket exit
# ---------------------------------------------------------------------------


def test_websocket_send_serialized_follows_stack_top_as_text_frame():
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_serialized(
            {"name": "before", "nickname": None},
            response_model=Item,
        )
        push_response_policy(exclude_none=True)
        await websocket.send_serialized(
            {"name": "after", "nickname": None},
            response_model=Item,
        )
        await websocket.close()

    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        first = websocket.receive_json()
        second = websocket.receive_json()
    assert first == {"NAMED": "before", "nickname": None, "price": 10.0}
    assert second == {"NAMED": "after", "price": 10.0}


def test_websocket_send_serialized_without_model_uses_switches():
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        push_response_policy(exclude_none=True)
        await websocket.send_serialized({"name": "x", "nickname": None})
        await websocket.close()

    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        # WebSocket text frame: no HTTP content-type/media type involved.
        text = websocket.receive_text()
    assert text == json.dumps({"name": "x"})


def test_websocket_strict_failure_can_be_written_to_close_reason():
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await websocket.send_serialized(
                {"name": 123},
                response_model=Item,
            )
        except ResponseValidationError as exc:
            await websocket.close_validation_error(exc)
            return
        await websocket.close()

    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            websocket.receive_json()
    assert exc_info.value.code == 1011
    reason = json.loads(exc_info.value.reason)
    assert isinstance(reason, list)
    # Validation loc uses the field name, and jsonable_encoder turns the
    # loc tuple into a list.
    assert any(error["loc"] == ["response", "name"] for error in reason)


def test_websocket_lenient_mode_skips_validation_on_structured_payload():
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        push_response_policy(exclude_none=True, validation_mode=ValidationMode.lenient)
        await websocket.send_serialized(
            {"name": 123, "nickname": None},
            response_model=Item,
        )
        await websocket.close()

    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        payload = websocket.receive_json()
    assert payload["NAMED"] == 123
    assert "nickname" not in payload


def test_websocket_input_validation_handler_unchanged():
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket, q: int) -> None:
        await websocket.accept()
        await websocket.close()

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc_info:
        # The request-validation failure closes the connection during the
        # handshake, before any frame could be pushed.
        with client.websocket_connect("/ws?q=not-an-int"):
            pass  # pragma: no cover
    assert exc_info.value.code == 1008
    # The built-in handler passes the encoded error list as the reason; the
    # test transport may hand it back decoded.
    raw_reason = exc_info.value.reason
    errors = raw_reason if isinstance(raw_reason, list) else json.loads(raw_reason)
    assert any(error["loc"][-1] == "q" for error in errors)


def test_concurrent_websocket_connections_have_independent_stacks():
    app = FastAPI()
    barrier = threading.Barrier(2, timeout=10)

    @app.websocket("/ws/{who}")
    async def ws_endpoint(websocket: WebSocket, who: str) -> None:
        await websocket.accept()
        if who == "a":
            push_response_policy(exclude_none=True)
        else:
            push_response_policy(by_alias=False)
        # Force the two connections (running on separate client portals/loops)
        # to overlap while both frames are pushed.
        await anyio.to_thread.run_sync(barrier.wait)
        await websocket.send_serialized(
            {"name": who, "nickname": None}, response_model=Item
        )
        await websocket.close()

    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def talk(who: str) -> None:
        try:
            client = TestClient(app)
            with client.websocket_connect(f"/ws/{who}") as websocket:
                results[who] = websocket.receive_json()
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    first = threading.Thread(target=talk, args=("a",))
    second = threading.Thread(target=talk, args=("b",))
    first.start()
    second.start()
    first.join()
    second.join()
    assert errors == []
    assert results["a"] == {"NAMED": "a", "price": 10.0}
    assert results["b"] == {"name": "b", "nickname": None, "price": 10.0}


def test_websocket_frame_does_not_leak_to_later_connection():
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket, push: bool = False) -> None:
        await websocket.accept()
        if push:
            push_response_policy(exclude_none=True)  # never popped on purpose
        await websocket.send_serialized({"name": "x", "nickname": None})
        await websocket.close()

    client = TestClient(app)
    with client.websocket_connect("/ws?push=true") as websocket:
        assert websocket.receive_json() == {"name": "x"}
    # A later connection on the same app starts with a clean stack.
    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {"name": "x", "nickname": None}


# ---------------------------------------------------------------------------
# 6. Default exception handlers do not subscribe to the stack
# ---------------------------------------------------------------------------


def test_default_http_exception_handler_keeps_detail_json_under_frame():
    app = FastAPI()

    def push_frame() -> None:
        push_response_policy(exclude_none=True, by_alias=False)

    @app.get("/boom", dependencies=[Depends(push_frame)])
    async def boom() -> None:
        raise HTTPException(status_code=404, detail={"msg": None, "code": 1})

    @app.get("/boom-plain")
    async def boom_plain() -> None:
        raise HTTPException(status_code=404, detail={"msg": None, "code": 1})

    client = TestClient(app)
    pushed = client.get("/boom")
    plain = client.get("/boom-plain")
    assert pushed.status_code == 404
    assert pushed.headers["content-type"] == "application/json"
    assert pushed.json() == {"detail": {"msg": None, "code": 1}}
    # Identical to the request that never pushed anything.
    assert pushed.json() == plain.json()


def test_default_request_validation_error_handler_keeps_detail_under_frame():
    app = FastAPI()

    def push_frame() -> None:
        push_response_policy(exclude_none=True, by_alias=False)

    @app.get("/check")
    async def check(
        q: Annotated[int, Query()],
        _: Annotated[None, Depends(push_frame)] = None,
    ) -> int:
        return q

    @app.get("/check-plain")
    async def check_plain(q: Annotated[int, Query()]) -> int:
        return q

    client = TestClient(app)
    pushed = client.get("/check?q=not-an-int")
    plain = client.get("/check-plain?q=not-an-int")
    assert pushed.status_code == 422
    assert pushed.json() == plain.json()
    detail = pushed.json()["detail"]
    assert detail[0]["loc"] == ["query", "q"]


# ---------------------------------------------------------------------------
# 7. OpenAPI / JSON Schema only read decorator fields
# ---------------------------------------------------------------------------


def test_openapi_ignores_request_time_stack_frames():
    app = FastAPI()

    def weird_frames() -> None:
        push_response_policy(exclude_none=True, by_alias=False)

    @app.get(
        "/item",
        response_model=Item,
        response_model_exclude_none=True,
        dependencies=[Depends(weird_frames)],
    )
    async def get_item() -> Any:
        return Item(name="x")

    client = TestClient(app)
    before = dict(app.openapi())
    client.get("/item")
    # The document is identical before and after a request that pushed frames.
    assert app.openapi() == before
    # And the schema itself is generated from model/decorator metadata: the
    # alias is present in JSON Schema regardless of request-time by_alias.
    schema = before["components"]["schemas"]["Item"]
    assert "NAMED" in schema["properties"]
    assert before["paths"]["/item"]["get"]["responses"]["200"] is not None


# ---------------------------------------------------------------------------
# 8. BackgroundTasks observe an empty stack
# ---------------------------------------------------------------------------


def test_background_tasks_run_with_empty_stack_view():
    app = FastAPI()
    seen: list[Any] = []

    def background_job() -> None:
        seen.append((get_pushed_response_policy(),))

    @app.get("/item")
    async def get_item(background_tasks: BackgroundTasks) -> dict[str, str]:
        push_response_policy(exclude_none=True)  # never popped on purpose
        background_tasks.add_task(background_job)
        return {"status": "scheduled"}

    client = TestClient(app)
    response = client.get("/item")
    assert response.status_code == 200
    assert seen == [(None,)]

    # The leaked frame did not survive into the next request either.
    response = client.get("/item")
    assert response.status_code == 200
    assert all(item == (None,) for item in seen)


# ---------------------------------------------------------------------------
# 9. Concurrency isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_concurrent_asgi_requests_keep_their_own_stack_tops():
    app = FastAPI()
    a_pushed = anyio.Event()
    release = anyio.Event()

    @app.get("/a", response_model=Item)
    async def endpoint_a() -> Any:
        push_response_policy(exclude_none=True)
        a_pushed.set()
        await release.wait()
        return Item(name="a", nickname=None)

    @app.get("/b", response_model=Item)
    async def endpoint_b() -> Any:
        push_response_policy(by_alias=False)
        await a_pushed.wait()
        release.set()
        return Item(name="b", nickname=None)

    results: dict[str, dict[str, Any]] = {}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:

        async def get(who: str) -> None:
            results[who] = (await client.get(f"/{who}")).json()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(get, "a")
            task_group.start_soon(get, "b")

    # A only sees its exclude_none frame; B only sees its by_alias frame,
    # even though both requests overlap with frames pushed.
    assert results["a"] == {"NAMED": "a", "price": 10.0}
    assert results["b"] == {"name": "b", "nickname": None, "price": 10.0}


@pytest.mark.anyio
async def test_streaming_request_isolation_while_other_request_pushes_pops():
    app = FastAPI()

    @app.get("/stream")
    async def stream():
        push_response_policy(exclude_none=True)
        for index in range(3):
            # Self-check: B's frames must never appear on A's request stack.
            folded = get_pushed_response_policy()
            assert folded is not None
            assert folded.exclude_none is True
            assert folded.by_alias is None
            yield {"name": "a", "nickname": None, "index": index}
            await anyio.sleep(0.02)

    @app.get("/noisy", response_model=Item)
    async def noisy() -> Any:
        # Raw push (not the context manager): the frame must still be on the
        # stack when the framework serializes this response after it returns.
        push_response_policy(by_alias=False)
        for _ in range(5):
            folded = get_pushed_response_policy()
            assert folded.by_alias is False
            await anyio.sleep(0.005)
        return Item(name="noisy", nickname=None)

    stream_lines: list[dict[str, Any]] = []
    noisy_body: dict[str, Any] = {}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:

        async def consume_stream() -> None:
            async with client.stream("GET", "/stream") as response:
                async for line in response.aiter_lines():
                    if line:
                        stream_lines.append(json.loads(line))

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(consume_stream)
            await anyio.sleep(0.01)
            task_group.start_soon(lambda: _store_noisy(client, noisy_body))

    assert stream_lines == [{"name": "a", "index": index} for index in range(3)]
    # B's own top was honored for B.
    assert noisy_body == {"name": "noisy", "nickname": None, "price": 10.0}


async def _store_noisy(client: AsyncClient, target: dict[str, Any]) -> None:
    response = await client.get("/noisy")
    target.update(response.json())


def test_concurrent_testclient_threads_keep_isolated_stacks():
    app = FastAPI()
    barrier = threading.Barrier(2, timeout=10)

    @app.get("/a", response_model=Item)
    def endpoint_a() -> Any:
        import time

        push_response_policy(exclude_none=True)
        barrier.wait()
        time.sleep(0.01)
        return Item(name="a", nickname=None)

    @app.get("/b", response_model=Item)
    def endpoint_b() -> Any:
        import time

        push_response_policy(by_alias=False)
        barrier.wait()
        time.sleep(0.01)
        return Item(name="b", nickname=None)

    results: dict[str, dict[str, Any]] = {}
    errors: list[BaseException] = []

    def hit(who: str) -> None:
        try:
            client = TestClient(app)
            results[who] = client.get(f"/{who}").json()
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    first = threading.Thread(target=hit, args=("a",))
    second = threading.Thread(target=hit, args=("b",))
    first.start()
    second.start()
    first.join()
    second.join()
    assert errors == []
    assert results["a"] == {"NAMED": "a", "price": 10.0}
    assert results["b"] == {"name": "b", "nickname": None, "price": 10.0}


# ---------------------------------------------------------------------------
# 10. dependency_overrides coexistence
# ---------------------------------------------------------------------------


def test_policy_stack_and_dependency_overrides_restore_independently():
    app = FastAPI()

    def policy_dependency() -> None:
        push_response_policy(exclude_none=True)

    @app.get(
        "/item",
        response_model=Item,
        dependencies=[Depends(policy_dependency)],
    )
    async def get_item() -> Any:
        return Item(name="x", nickname=None)

    client = TestClient(app)
    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}

    # Override the dependency with one that pushes a different frame; the
    # override does not require clearing the stack first.
    app.dependency_overrides[policy_dependency] = lambda: push_response_policy(
        by_alias=False
    )
    try:
        assert client.get("/item").json() == {
            "name": "x",
            "nickname": None,
            "price": 10.0,
        }
    finally:
        # Clearing the override does not need to pop anything, and vice versa.
        app.dependency_overrides.clear()

    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}
    assert get_pushed_response_policy() is None


# ---------------------------------------------------------------------------
# 11. Middleware frames flow into the downstream request stack
# ---------------------------------------------------------------------------


def test_base_http_middleware_frame_inherited_by_request():
    from starlette.middleware.base import BaseHTTPMiddleware

    class DropNoneMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):  # type: ignore[override]
            with response_policy(exclude_none=True):
                return await call_next(request)

    app = FastAPI()
    app.add_middleware(DropNoneMiddleware)

    @app.get("/item", response_model=Item)
    async def get_item() -> Any:
        return Item(name="x", nickname=None)

    client = TestClient(app)
    assert client.get("/item").json() == {"NAMED": "x", "price": 10.0}


# ---------------------------------------------------------------------------
# 12. Old routes never push anything: response_model_* keeps current behavior
# ---------------------------------------------------------------------------


def test_old_routes_keep_decorator_only_behavior():
    app = FastAPI()

    @app.get("/default", response_model=Item)
    async def default_item() -> Any:
        return Item(name="x", nickname=None)

    @app.get(
        "/exclude-none",
        response_model=Item,
        response_model_exclude_none=True,
    )
    async def exclude_none_item() -> Any:
        return Item(name="x", nickname=None)

    @app.get(
        "/flat",
        response_model=Item,
        response_model_by_alias=False,
    )
    async def flat_item() -> Any:
        return Item(name="x", nickname=None)

    @app.get("/router-prefix", response_model=Item)
    async def router_item() -> Any:
        return Item(name="r", nickname=None)

    router = APIRouter()
    router.add_api_route(
        "/r",
        router_item,
        response_model=Item,
        response_model_exclude_none=True,
    )
    app.include_router(router, prefix="/router-prefix")

    client = TestClient(app)
    assert client.get("/default").json() == {
        "NAMED": "x",
        "nickname": None,
        "price": 10.0,
    }
    assert client.get("/exclude-none").json() == {"NAMED": "x", "price": 10.0}
    assert client.get("/flat").json() == {
        "name": "x",
        "nickname": None,
        "price": 10.0,
    }
    assert client.get("/router-prefix/r").json() == {
        "NAMED": "r",
        "price": 10.0,
    }
