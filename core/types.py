# neron_llm/core/types.py
# Data models for neron_llm — standardized request/response formats.

from __future__ import annotations
from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field

# ── Limits ────────────────────────────────────────────────────────────────────

# ~32k chars ≈ ~8k tokens — sufficient for large code reviews / long prompts
PROMPT_MAX_LEN        = 32_768

# Reasonable ceiling for the legacy /chat message field
MESSAGE_MAX_LEN       = 32_768

# Context dict: max number of keys and max length per value
CONTEXT_MAX_KEYS      = 20
CONTEXT_VALUE_MAX_LEN = 2_048


# ── Internal types (LLMManager / providers) ───────────────────────────────────

class LLMRequest(BaseModel):
    """Internal request passed through Manager → Provider pipeline."""

    message:  str = Field(..., min_length=1, max_length=MESSAGE_MAX_LEN)
    task:     Optional[str]                              = Field(default=None)
    mode:     Optional[Literal["single", "parallel", "race"]] = Field(default=None)
    provider: Optional[str]                              = Field(default=None)
    model:    Optional[str]                              = Field(default=None)
    metadata: Optional[Dict[str, str]]                   = Field(default=None)
    json_mode: bool                                      = Field(default=False)


class LLMResponse(BaseModel):
    """Internal response from the Manager pipeline."""

    model:    str
    provider: str
    response: str
    error:    Optional[str] = None


# ── Public bus contract ───────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Payload received at POST /llm/generate — the only external contract."""

    task_type:        Literal["code", "reasoning", "chat", "agent", "memory", "fast", "summary", "default"] = Field(default="chat")
    prompt:           str             = Field(..., min_length=1, max_length=PROMPT_MAX_LEN)
    context:          Dict[str, str]  = Field(default_factory=dict, max_length=CONTEXT_MAX_KEYS)
    model_preference: str             = Field(default="auto")
    request_id:       str             = Field(default="")
    json_mode:        bool            = Field(default=False)


class GenerateResponse(BaseModel):
    """Response returned by POST /llm/generate."""

    result:     str
    model_used: str
    latency_ms: int
    warning:    Optional[str] = None
