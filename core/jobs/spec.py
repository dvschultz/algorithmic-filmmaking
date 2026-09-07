"""Immutable JSON operation metadata, independent of Qt and media backends."""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal, cast


def encode_object(value: dict) -> str:
    """Detach a JSON object without coercing keys or non-JSON Python objects."""

    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Operation object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("Operation values must be JSON data")

    if not isinstance(value, dict):
        raise ValueError("Operation metadata must be a JSON object")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class OperationSpec:
    kind: str
    version: int
    arguments_json: str
    inputs_json: str
    persistence: Literal["job_history", "session_only"]
    cancellable: bool = True
    session_id: str | None = None
    input_revision: str | None = None
    publication: Literal["worker", "owner_thread"] = "worker"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Operation kind must be nonempty")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Operation version must be a positive integer")
        if (
            self.persistence not in ("job_history", "session_only")
            or type(self.cancellable) is not bool
            or self.publication not in ("worker", "owner_thread")
        ):
            raise ValueError("Invalid execution capabilities")
        for value in (self.session_id, self.input_revision):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    "Session and revision identities must be nonempty strings"
                )
        for value in (self.arguments_json, self.inputs_json):
            if encode_object(json.loads(value)) != value:
                raise ValueError("Operation JSON must be canonical")

    @classmethod
    def build(
        cls,
        *,
        kind: str,
        version: int,
        arguments: dict,
        inputs: dict,
        persistence: Literal["job_history", "session_only"],
        cancellable: bool = True,
        session_id: str | None = None,
        input_revision: str | None = None,
        publication: Literal["worker", "owner_thread"] = "worker",
    ) -> "OperationSpec":
        return cls(
            kind=kind,
            version=version,
            arguments_json=encode_object(arguments),
            inputs_json=encode_object(inputs),
            persistence=persistence,
            cancellable=cancellable,
            session_id=session_id,
            input_revision=input_revision,
            publication=publication,
        )

    @property
    def arguments(self) -> dict:
        return cast(dict, json.loads(self.arguments_json))

    def to_json(self) -> str:
        return encode_object(
            {
                "kind": self.kind,
                "version": self.version,
                "arguments": self.arguments,
                "inputs": json.loads(self.inputs_json),
                "persistence": self.persistence,
                "cancellable": self.cancellable,
                "session_id": self.session_id,
                "input_revision": self.input_revision,
                **(
                    {"publication": self.publication}
                    if self.publication != "worker"
                    else {}
                ),
            }
        )

    @classmethod
    def from_json(cls, value: str) -> "OperationSpec":
        return cls.build(**json.loads(value))

    @property
    def operation_id(self) -> str:
        return sha256(self.to_json().encode()).hexdigest()

    def safe_projection(self) -> dict:
        return {
            "id": self.operation_id,
            "kind": self.kind,
            "version": self.version,
            "persistence": self.persistence,
            "cancellable": self.cancellable,
            **(
                {"publication": self.publication}
                if self.publication != "worker"
                else {}
            ),
        }
