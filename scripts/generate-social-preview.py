"""Generate the checked-in share card; Pillow is only needed to regenerate it."""

from pathlib import Path
import os

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/dejavu")]


def font(size, bold=False):
    names = ["segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "DejaVuSans.ttf"]
    for directory in FONT_DIRS:
        for name in names:
            path = directory / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    raise RuntimeError("Install Segoe UI or DejaVu Sans to regenerate the share card")


if __name__ == "__main__":
    image = Image.new("RGB", (1200, 630), "#081218")
    draw = ImageDraw.Draw(image)
    draw.rectangle((72, 100, 132, 108), fill="#f2a13c")
    draw.text((72, 147), "RAGnarok", font=font(100, True), fill="#f4efe2")
    draw.text((76, 294), "Manga & anime discovery", font=font(44), fill="#5cd2b1")
    draw.text((76, 368), "Find a story. Paste a chapter URL. Read.", font=font(35), fill="#b9c2c5")
    draw.text((76, 525), "ragnarok-reader.vercel.app", font=font(27), fill="#b9c2c5")
    image.save(ROOT / "apps/frontend/public/social-preview.png", optimize=True)
