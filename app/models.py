"""Request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from . import config


class Word(BaseModel):
    text: str = Field(min_length=1, description="token text")
    start_ms: int = Field(ge=0, description="word start time in milliseconds")
    end_ms: int = Field(gt=0, description="word end time in milliseconds")

    @model_validator(mode="after")
    def _check_times(self) -> "Word":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class SegmentRequest(BaseModel):
    words: list[Word] = Field(
        min_length=1,
        description="time-ordered, non-overlapping word-level transcript",
    )
    max_chars: int = Field(
        default=config.DEFAULT_MAX_CHARS,
        ge=1,
        description="maximum number of characters per subtitle",
    )
    min_duration_ms: int = Field(
        default=config.DEFAULT_MIN_DURATION_MS,
        ge=0,
        description="minimum subtitle duration in milliseconds",
    )
    max_duration_ms: int = Field(
        default=config.DEFAULT_MAX_DURATION_MS,
        gt=0,
        description="maximum subtitle duration in milliseconds",
    )
    target_duration_ms: int = Field(
        default=config.DEFAULT_TARGET_DURATION_MS,
        ge=0,
        description="ideal subtitle duration in milliseconds",
    )
    pause_threshold_ms: int = Field(
        default=config.DEFAULT_PAUSE_THRESHOLD_MS,
        ge=0,
        description="gap (ms) between words that counts as a long pause",
    )
    max_line_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "optional line width for at-most-two-line typesetting: when set, "
            "each subtitle is laid out on one or two lines that break only "
            "between words, with every line no longer than this many "
            "characters (joining spaces included); omitted to keep plain "
            "single-line output"
        ),
    )
    required_break_after: list[int] | None = Field(
        default=None,
        description=(
            "optional manual editing boundaries: zero-based indices into "
            "this request's words after which a subtitle boundary must be "
            "placed; each index must be between 0 and len(words)-2 and must "
            "not fall inside a keep_together interval"
        ),
    )
    keep_together: list[list[int]] | None = Field(
        default=None,
        description=(
            "optional manual editing boundaries: closed [start, end] "
            "zero-based word-index intervals (each spanning at least two "
            "words) that no subtitle boundary may fall inside; intervals "
            "must not overlap one another and must not contain a "
            "required_break_after index"
        ),
    )

    @model_validator(mode="after")
    def _check_limits(self) -> "SegmentRequest":
        if self.max_duration_ms < self.min_duration_ms:
            raise ValueError("max_duration_ms must be >= min_duration_ms")
        return self


class SegmentOut(BaseModel):
    index: int
    text: str
    start_ms: int
    end_ms: int
    duration_ms: int
    word_start: int
    word_end: int
    crossed_boundaries: int


class SegmentResponse(BaseModel):
    segment_count: int
    total_cost: float
    segments: list[SegmentOut]
