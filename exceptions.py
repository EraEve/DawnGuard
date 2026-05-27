"""Custom exception hierarchy for the MAPFM medical AI ecosystem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MAPFMException(Exception):
    """Base exception with a stable error code.

    Args:
        message: Human-readable error description.
        code: Machine-readable error code.
        cause: Optional chained exception.
    """

    message: str
    code: str = "MAPFM_ERROR"
    cause: Optional[BaseException] = None

    def __post_init__(self) -> None:
        super().__init__(f"[{self.code}] {self.message}")

    def to_dict(self) -> dict:
        """Return a serializable representation of the exception."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "cause": repr(self.cause) if self.cause is not None else None,
        }


class ConfigError(MAPFMException):
    """Raised when configuration is invalid."""


class RetrievalError(MAPFMException):
    """Raised when retrieval or index synchronization fails."""


class InferenceError(MAPFMException):
    """Raised when DMA inference or structured-output parsing fails."""


class AgentError(MAPFMException):
    """Raised when an agent cannot complete its required interface contract."""


class DatasetError(MAPFMException):
    """Raised when dataset loading or validation fails."""
