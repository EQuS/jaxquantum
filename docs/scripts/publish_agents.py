"""Publish the repository's agent guide at stable site URLs."""

from pathlib import Path
from shutil import copyfile


def on_post_build(config):
    guide = Path(__file__).resolve().parents[2] / "AGENTS.md"
    site = Path(config["site_dir"])
    for name in ("AGENTS.md", "agents.md"):
        copyfile(guide, site / name)
