"""Smoke-test scene — used to verify render.sh works before real scenes exist.

Replaced in subsequent issues (07+) by actual content scenes.
"""

from manim import Scene, Text


class SmokeScene(Scene):
    """A single text mobject rendered for 1 second. Used as a render.sh canary."""

    def construct(self) -> None:
        msg = Text("Glassbox smoke test", font_size=48)
        self.add(msg)
        self.wait(1)
