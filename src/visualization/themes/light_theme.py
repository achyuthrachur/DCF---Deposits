"""Light theme configuration for Monte Carlo dashboards."""

from __future__ import annotations

from typing import Dict


PRIMARY_BLUE = "#2E86AB"
PRIMARY_NAVY = "#1F4788"
SECONDARY_TEAL = "#06A77D"
SECONDARY_CYAN = "#00D9FF"
POSITIVE_GREEN = "#4CAF50"
NEGATIVE_RED = "#FF5252"
WARNING_ORANGE = "#FF6F00"
NEUTRAL_GRAY = "#757575"
BACKGROUND = "#FAFAFA"
CARD_BACKGROUND = "#FFFFFF"
TEXT_COLOR = "#1E1E1E"
SUBTEXT_COLOR = "#424242"


LIGHT_THEME: Dict[str, object] = {
    "name": "light",
    "background_color": BACKGROUND,
    "card_background": CARD_BACKGROUND,
    "text_color": TEXT_COLOR,
    "subtext_color": SUBTEXT_COLOR,
    "palette": {
        "primary_blue": PRIMARY_BLUE,
        "primary_navy": PRIMARY_NAVY,
        "secondary_teal": SECONDARY_TEAL,
        "secondary_cyan": SECONDARY_CYAN,
        "positive": POSITIVE_GREEN,
        "negative": NEGATIVE_RED,
        "warning": WARNING_ORANGE,
        "neutral": NEUTRAL_GRAY,
    },
    "plotly_template": {
        "layout": {
            "font": {"family": "Roboto, Open Sans, sans-serif", "color": TEXT_COLOR},
            "paper_bgcolor": BACKGROUND,
            "plot_bgcolor": CARD_BACKGROUND,
            "title": {"font": {"size": 22, "color": TEXT_COLOR}},
            "legend": {"bgcolor": CARD_BACKGROUND, "bordercolor": "#E0E0E0"},
            "xaxis": {
                "gridcolor": "#E0E0E0",
                "linecolor": "#BDBDBD",
                "zerolinecolor": "#E0E0E0",
            },
            "yaxis": {
                "gridcolor": "#E0E0E0",
                "linecolor": "#BDBDBD",
                "zerolinecolor": "#E0E0E0",
            },
        }
    },
}
