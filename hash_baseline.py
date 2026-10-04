"""Deterministic, model-free image lookup using shape and colour."""

from pathlib import Path

import numpy as np
from PIL import Image

from vision import canonical, read_image


def dhash(image):
    """Return a 64-bit difference hash after the same fixed crop as the main method."""
    small = canonical(image).convert('L').resize((9, 8), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.uint8)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flat
    return sum(int(bit) << index for index, bit in enumerate(bits))


def colour_histogram(image):
    pixels = np.asarray(canonical(image), dtype=np.uint8).reshape(-1, 3)
    foreground = np.min(pixels, axis=1) < 245
    selected = pixels[foreground] if np.any(foreground) else pixels
    hist, _ = np.histogramdd(selected, bins=(4, 4, 4), range=((0, 256),) * 3)
    return (hist / hist.sum()).ravel().astype('float32')


class HandcraftedBaseline:
    def __init__(self, rows, gallery_dir):
        self.rows = rows
        gallery_dir = Path(gallery_dir)
        images = [read_image(gallery_dir / f"{row['id']}.jpg") for row in rows]
        self.hashes = [dhash(image) for image in images]
        self.histograms = np.stack([colour_histogram(image) for image in images])

    def rank(self, image, k=5):
        query = dhash(image)
        histogram = colour_histogram(image)
        colour = np.minimum(self.histograms, histogram).sum(axis=1)
        scores = [0.6 * (1 - (query ^ value).bit_count() / 64) + 0.4 * float(colour[index])
                  for index, value in enumerate(self.hashes)]
        order = sorted(range(len(self.rows)), key=lambda index: (-scores[index], self.rows[index]['id']))[:k]
        return [{'record': self.rows[index],
                 'score': round(scores[index], 6)}
                for index in order]
