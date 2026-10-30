"""Generate the BDDL task file for the LIBERO drawing problem."""
from __future__ import annotations

from pathlib import Path

from burger.backends.libero.environment import ENV_NAME, SceneLayout

BDDL_TEMPLATE = """(define (problem {problem})
  (:domain robosuite)
  (:language {language})
    (:regions
      (canvas_region
          (:target main_table)
          (:ranges (
              ({x0:.4f} {y0:.4f} {x1:.4f} {y1:.4f})
            )
          )
      )
      (canvas_surface_region
          (:target drawing_canvas_1)
      )
    )

  (:fixtures
    main_table - table
    drawing_canvas_1 - drawing_canvas
  )

  (:objects
  )

  (:obj_of_interest
    drawing_canvas_1
  )

  (:init
    (On drawing_canvas_1 main_table_canvas_region)
  )

  (:goal
    (And (On drawing_canvas_1 main_table_canvas_region))
  )

)
"""


def render_bddl(language: str, layout: SceneLayout, region_half_width: float = 0.001) -> str:
    """BDDL text for a drawing task. The canvas region is expressed relative to the LIBERO
    table-top centre (``workspace_offset``); the env re-applies the exact pose on reset."""
    cx = layout.canvas_center_xyz[0] - layout.workspace_offset[0]
    cy = layout.canvas_center_xyz[1] - layout.workspace_offset[1]
    if not isinstance(language, str) or not language.strip():
        raise ValueError("BDDL language must be a nonempty string")
    if any(character in language for character in "();\n\r\t") or any(ord(character) < 32 for character in language):
        raise ValueError("BDDL language cannot contain parentheses, semicolons or control characters")
    return BDDL_TEMPLATE.format(
        problem=ENV_NAME.upper(),
        language=language,
        x0=cx - region_half_width, y0=cy - region_half_width,
        x1=cx + region_half_width, y1=cy + region_half_width,
    )


def write_bddl(path: str | Path, language: str, layout: SceneLayout) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_bddl(language, layout))
    return path
