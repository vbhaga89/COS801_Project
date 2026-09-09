from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = ("healthy", "leaf_spot", "blight")
COUNTS = {"train": 24, "val": 9, "test": 9}


def make_image(label: str, index: int, size: int, seed: int) -> Image.Image:
    rng = random.Random(f"{seed}:{label}:{index}")
    image = Image.new("RGB", (size, size), (224, 220, 190))
    draw = ImageDraw.Draw(image)
    margin = size // 8
    green = (45 + rng.randrange(30), 125 + rng.randrange(45), 45 + rng.randrange(30))
    draw.ellipse((margin, margin // 2, size - margin, size - margin // 2), fill=green)
    draw.line((size // 2, margin, size // 2, size - margin), fill=(220, 235, 190), width=max(1, size // 40))

    if label == "leaf_spot":
        for _ in range(8):
            radius = rng.randrange(max(2, size // 30), max(3, size // 12))
            x = rng.randrange(margin + radius, size - margin - radius)
            y = rng.randrange(margin + radius, size - margin - radius)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(105, 62, 30))
    elif label == "blight":
        draw.pieslice(
            (margin, margin // 2, size - margin, size - margin // 2),
            start=rng.randrange(160, 210),
            end=rng.randrange(260, 330),
            fill=(180, 130, 40),
        )

    pixels = image.load()
    for _ in range(size * 2):
        x, y = rng.randrange(size), rng.randrange(size)
        red, green_channel, blue = pixels[x, y]
        delta = rng.randrange(-8, 9)
        pixels[x, y] = tuple(max(0, min(255, value + delta)) for value in (red, green_channel, blue))
    return image


def build_dataset(output: Path, size: int, seed: int) -> None:
    for split, count in COUNTS.items():
        for label in CLASSES:
            class_dir = output / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                path = class_dir / f"{label}_{split}_{index:03d}.png"
                make_image(label, index, size, seed).save(path)
    marker = output / "SYNTHETIC_SMOKE_DATA_ONLY.txt"
    marker.write_text(
        "Synthetic images for pipeline validation only. Never use these data as research results.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the deterministic RQ7.1 smoke-test dataset")
    parser.add_argument("--output", default="artifacts/smoke_data")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)
    build_dataset(output, args.size, args.seed)
    print(f"Created smoke dataset at {output.resolve()}")


if __name__ == "__main__":
    main()
