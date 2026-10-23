import json
from typing import Any

from fastapi._compat import ModelField
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import ResponseValidationError
from fastapi.response_policy import (
    ValidationMode,
    best_effort_construct,
    lenient_serialization,
    resolve_response_policy,
)
from fastapi.types import IncEx
from fastapi.utils import create_model_field
from starlette import status
from starlette.websockets import WebSocket as StarletteWebSocket
from starlette.websockets import WebSocketDisconnect as WebSocketDisconnect  # noqa
from starlette.websockets import WebSocketState as WebSocketState  # noqa

# WebSocket close reasons are limited to 123 bytes by the protocol.
_MAX_WS_CLOSE_REASON_BYTES = 123


class WebSocket(StarletteWebSocket):
    """FastAPI WebSocket connection.

    Beyond the Starlette connection, it offers the WebSocket response
    serialization exit point: :meth:`send_serialized` validates and encodes a
    payload following the current request (connection) response serialization
    policy stack, and writes it as a WebSocket **text message**. It never goes
    through HTTP ``JSONResponse`` / media type assembly.
    """

    def serialize_message(
        self,
        data: Any,
        *,
        response_model: Any = None,
        field: ModelField | None = None,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        by_alias: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> str:
        """Serialize ``data`` to a JSON text payload for a WebSocket message.

        The keyword arguments are the connection/route baseline; the current
        policy stack top is merged with them at call time. With
        ``validation_mode="strict"`` (the default) and a ``response_model``,
        a bad payload raises :class:`fastapi.exceptions.ResponseValidationError`;
        with ``"lenient"`` the validation pass is skipped and the raw payload
        is serialized, still honoring the other switches.

        This is a WebSocket-only exit: it produces JSON text and never builds
        an HTTP response or media type.
        """
        effective = resolve_response_policy(
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        model_field = field
        if model_field is None and response_model is not None:
            model_field = create_model_field(
                name="WebSocketResponse",
                type_=response_model,
                mode="serialization",
            )
        if model_field is not None:
            if effective.validation_mode is ValidationMode.strict:
                value, errors = model_field.validate(data, {}, loc=("response",))
                if errors:
                    raise ResponseValidationError(errors=errors, body=data)
            else:
                value = best_effort_construct(model_field, data)
            if effective.validation_mode is ValidationMode.lenient:
                with lenient_serialization():
                    encoded = model_field.serialize(
                        value,
                        include=include,
                        exclude=exclude,
                        **effective.dump_kwargs,
                    )
            else:
                encoded = model_field.serialize(
                    value,
                    include=include,
                    exclude=exclude,
                    **effective.dump_kwargs,
                )
        else:
            encoded = jsonable_encoder(
                data,
                include=include,
                exclude=exclude,
                **effective.dump_kwargs,
            )
        return json.dumps(encoded)

    async def send_serialized(
        self,
        data: Any,
        *,
        response_model: Any = None,
        field: ModelField | None = None,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        by_alias: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> None:
        """Validate/serialize ``data`` following the stack top and send it.

        The payload is sent as a WebSocket text frame (JSON), not through any
        HTTP media type machinery. The stack is read at call time, so frames
        pushed or popped between messages inside the connection loop only
        affect later messages.
        """
        text = self.serialize_message(
            data,
            response_model=response_model,
            field=field,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        await self.send_text(text)

    async def close_validation_error(
        self,
        exc: ResponseValidationError,
        *,
        code: int = status.WS_1011_INTERNAL_ERROR,
    ) -> None:
        """Close the connection putting a validation failure in the reason.

        The structured error list is encoded following the serialization
        switches of the current stack top (validation failures only occur
        under ``validation_mode="strict"``) and the connection is closed with
        ``code`` (1011 internal error by default). The reason is truncated to
        the WebSocket protocol limit of 123 bytes.
        """
        effective = resolve_response_policy()
        errors = jsonable_encoder(exc.errors(), **effective.dump_kwargs)
        reason = json.dumps(errors)
        encoded = reason.encode("utf-8")[:_MAX_WS_CLOSE_REASON_BYTES]
        await self.close(code=code, reason=encoded.decode("utf-8", "ignore"))
