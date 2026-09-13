from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import BY_ID


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    task: str
    models: list[Literal["flash", "base"]] = Field(default_factory=lambda: ["flash"], min_length=1, max_length=2)
    source_id: str | None = None
    text: str = Field(default="", max_length=6000)
    instruction: str = Field(default="", max_length=6000)
    duration: float | None = Field(default=None, ge=0.02, le=30)
    speed_factor: float = Field(default=1.25, ge=0.5, le=2)
    seed: int = Field(default=1234, ge=0, le=2**32 - 6)
    repeats: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def validate_task(self):
        if self.task not in BY_ID:
            raise ValueError("Unknown task")
        task = BY_ID[self.task]
        if len(set(self.models)) != len(self.models):
            raise ValueError("Choose each model only once")
        if task["audio"] and not self.source_id:
            raise ValueError("This task requires source audio")
        if self.task == "voice_design" and self.source_id:
            raise ValueError("Voice design does not accept reference audio; use voice cloning or free-form")
        if task["route"] == "speech" and not self.text.strip():
            raise ValueError("Enter the text to speak")
        if task["route"] == "edit" and not self.instruction.strip():
            raise ValueError("Enter an instruction")
        if self.task in {"voice_design", "voice_clone"} or not self.source_id:
            if self.duration is None:
                raise ValueError("Set an explicit output duration for synthesis")
        return self


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["unrated", "keep", "mixed", "reject"] = "unrated"
    content: int | None = Field(default=None, ge=1, le=5)
    identity: int | None = Field(default=None, ge=1, le=5)
    instruction: int | None = Field(default=None, ge=1, le=5)
    quality: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=6000)
