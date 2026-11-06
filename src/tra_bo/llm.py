"""Acquisition selectors and an OpenAI-compatible LLM client."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
import json
import os
import re
import time
from typing import Any, Mapping, Protocol, Sequence
import urllib.error
import urllib.request

from .acquisition import ACQUISITIONS


TRA_BO_SYSTEM_PROMPT = resources.files("tra_bo.prompts").joinpath(
    "acquisition_controller.txt"
).read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class BOState:
    """State visible to the acquisition selector before a BO query."""

    tra: float
    tra_valid: bool
    evaluations: int
    initial_evaluations: int
    remaining_budget: int
    total_budget: int
    dimension: int
    incumbent: float
    incumbent_history: tuple[float, ...]
    target_mean: float
    target_std: float
    target_range: float
    latest_min_distance: float
    gp_uncertainty_mean: float
    gp_uncertainty_max: float
    gp_lengthscale_min: float
    gp_lengthscale_mean: float
    gp_lengthscale_max: float
    gp_output_scale: float
    previous_acquisition: str | None
    acquisition_history: tuple[str, ...]
    improvement_history: tuple[bool, ...]

    @property
    def progress(self) -> float:
        denominator = max(self.total_budget - self.initial_evaluations, 1)
        return float((self.evaluations - self.initial_evaluations) / denominator)

    def as_prompt_payload(self) -> dict[str, Any]:
        state = asdict(self)
        state["progress"] = self.progress
        non_improvement_streak = 0
        for improved in reversed(self.improvement_history):
            if improved:
                break
            non_improvement_streak += 1
        state["non_improvement_streak"] = non_improvement_streak
        state["action_usage"] = {
            action: self.acquisition_history.count(action) for action in ACQUISITIONS
        }
        state["action_successes"] = {
            action: sum(
                selected == action and improved
                for selected, improved in zip(
                    self.acquisition_history, self.improvement_history
                )
            )
            for action in ACQUISITIONS
        }
        return {
            "selection_protocol": "select_acquisition_then_optimize",
            "available_acquisitions": list(ACQUISITIONS),
            "state": state,
        }


class AcquisitionSelector(Protocol):
    """Interface implemented by acquisition policies."""

    def select(self, state: BOState) -> str:
        ...


class FixedAcquisitionSelector:
    """Fixed policy useful for ablations and tests."""

    def __init__(self, acquisition: str = "EI") -> None:
        action = acquisition.upper()
        if action not in ACQUISITIONS:
            raise ValueError(f"acquisition must be one of {ACQUISITIONS}")
        self.acquisition = action

    def select(self, state: BOState) -> str:
        del state
        return self.acquisition


@dataclass(frozen=True)
class LLMConfig:
    """OpenAI-compatible endpoint configuration."""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 64
    timeout_seconds: float = 120.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        key = os.environ.get("TRABO_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("set TRABO_API_KEY or OPENAI_API_KEY")
        base_url = os.environ.get("TRABO_BASE_URL") or os.environ.get(
            "OPENAI_BASE_URL", ""
        )
        if not base_url:
            raise ValueError("set TRABO_BASE_URL or OPENAI_BASE_URL")
        model = os.environ.get("TRABO_MODEL") or os.environ.get("OPENAI_MODEL", "")
        if not model:
            raise ValueError("set TRABO_MODEL or OPENAI_MODEL")
        return cls(
            api_key=key,
            base_url=base_url,
            model=model,
            temperature=float(os.environ.get("TRABO_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("TRABO_MAX_TOKENS", "64")),
            timeout_seconds=float(os.environ.get("TRABO_TIMEOUT", "120")),
            max_retries=int(os.environ.get("TRABO_MAX_RETRIES", "2")),
        )


class ChatClient(Protocol):
    """Minimal interface used by the LLM selector."""

    model: str

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        ...


class OpenAICompatibleClient:
    """Dependency-free client for OpenAI-compatible chat completions."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.model = self.config.model

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": list(messages),
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"])
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
                last_error = error
                if attempt < self.config.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError("LLM request failed") from last_error


class LLMAcquisitionSelector:
    """Select an acquisition function from the current TRA and BO state."""

    def __init__(
        self,
        client: ChatClient | None = None,
        fallback: str = "EI",
        system_prompt: str = TRA_BO_SYSTEM_PROMPT,
    ) -> None:
        self.client = client or OpenAICompatibleClient()
        self.fallback = fallback.upper()
        if self.fallback not in ACQUISITIONS:
            raise ValueError(f"fallback must be one of {ACQUISITIONS}")
        self.system_prompt = system_prompt
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def parse_acquisition(content: str) -> str | None:
        try:
            parsed = json.loads(content)
            action = str(parsed.get("acquisition", "")).upper()
            return action if action in ACQUISITIONS else None
        except (json.JSONDecodeError, AttributeError, TypeError):
            match = re.search(r"\b(PI|EI|UCB|TS)\b", content.upper())
            return match.group(1) if match else None

    def select(self, state: BOState) -> str:
        payload = state.as_prompt_payload()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
        started = time.perf_counter()
        error: str | None = None
        raw = ""
        try:
            raw = self.client.complete(messages)
            action = self.parse_acquisition(raw)
        except Exception as exception:
            action = None
            error = f"{type(exception).__name__}: {exception}"
        selected = action or self.fallback
        self.calls.append(
            {
                "model": self.client.model,
                "state": payload["state"],
                "response": raw,
                "acquisition": selected,
                "fallback_used": action is None,
                "error": error,
                "latency_seconds": time.perf_counter() - started,
            }
        )
        return selected
