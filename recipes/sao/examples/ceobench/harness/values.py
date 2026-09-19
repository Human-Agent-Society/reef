"""JSON values exchanged with the model, capture proxy and task container."""

JsonValue = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
