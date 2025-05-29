from __future__ import annotations
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

class DarkEvolveV2(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.blue,
        secondary_hue: colors.Color | str = colors.cyan,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_xxl,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Fira Code"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

        super().set(
            # Página
            body_background_fill="#0f172a",
            body_background_fill_dark="#0f172a",

            # Bloco (coluna com borda visível e sombra)
            block_background_fill="transparent",
            block_border_color="#1e293b",
            block_border_width="2px",
            block_shadow="0px 0px 10px rgba(0,0,0,0.3)",
            block_label_text_color="#e2e8f0",

            # Componentes internos (input, slider, etc)
            input_background_fill="rgba(255,255,255,0.05)",
            input_border_width="0px",
            input_shadow="none",
            
            # Botões
            button_primary_background_fill="#3b82f6",
            button_primary_background_fill_hover="#2563eb",
            button_primary_text_color="white",
            button_primary_border_color="transparent",
            button_primary_shadow="*shadow_drop_lg",

            # Tabs
          
            # Sliders
            slider_color="#3b82f6",
            slider_color_dark="#3b82f6",

            # Código
        )
