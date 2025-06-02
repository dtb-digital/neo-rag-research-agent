"""Configuration base class for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypedDict

from langchain_core.runnables import RunnableConfig


class SearchKwargs(TypedDict):
    """Search configuration."""

    k: int


@dataclass(kw_only=True)
class BaseConfiguration:
    """Base configuration for the agent."""

    embedding_model: str = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "The model to use for text embeddings. Should be in the form: provider/model-name."
        },
    )

    search_kwargs: SearchKwargs = field(
        default_factory=lambda: {"k": 5},
        metadata={
            "description": "Configuration for the search process, including number of documents to retrieve."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> BaseConfiguration:
        """Create a configuration from a RunnableConfig."""
        if not config or "configurable" not in config:
            return cls()
        return cls(**config["configurable"])
