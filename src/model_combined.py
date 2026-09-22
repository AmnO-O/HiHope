"""Compatibility module aliasing CombinedBackboneModel to TwoStreamBiEncoderModel."""
from __future__ import annotations
from src.model_two_stream import (
    TwoStreamBiEncoderModel,
    TwoStreamBiEncoderModel as CombinedBackboneModel,
)
from src.model import (
    build_model as build_two_stream_model,
    build_model as build_combined_model,
)

__all__ = [
    'TwoStreamBiEncoderModel',
    'CombinedBackboneModel',
    'build_two_stream_model',
    'build_combined_model',
]
