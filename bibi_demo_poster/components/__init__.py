"""UI Components package for IRONMAN 70.3 Demo."""
from .ui_blocks import (
    PALETTE,
    CHECKPOINT_LABELS,
    render_header_banner,
    render_checkpoint_stepper,
    render_hero_prediction_card,
    render_race_progress_card,
    make_live_comparison_figure,
    make_replay_chart,
    make_global_mae_chart,
)

__all__ = [
    "PALETTE",
    "CHECKPOINT_LABELS",
    "render_header_banner",
    "render_checkpoint_stepper",
    "render_hero_prediction_card",
    "render_race_progress_card",
    "make_live_comparison_figure",
    "make_replay_chart",
    "make_global_mae_chart",
]
