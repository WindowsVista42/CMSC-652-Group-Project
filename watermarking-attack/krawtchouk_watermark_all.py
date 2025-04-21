import os
import sys
import time
import glob

import numpy as np
from PIL import Image

import krawtchouk_watermark as kw

def watermark_a(path, in_path, bits, order, strength, p1, p2, nbits):
    comps = path.split(os.path.basename(os.path.normpath(in_path)))
    
    out_path = os.path.join("", *comps[:-1])
    out_path = os.path.join(out_path, f"watermark_{p1:.3f}_{p2:.3f}_{strength}")
    s = comps[-1].split(os.path.sep)
    out_path = os.path.join(out_path, *s)

    directory = os.path.dirname(out_path)
    if not os.path.exists(directory):
        print("Made directory:", directory)
        os.makedirs(directory, exist_ok=True)

    watermarked_image = kw.KM_watermark(path, bits, order, strength, p1, p2, nbits)
    watermarked_image.save(out_path)

if __name__ == "__main__":
    np.random.seed(42)
    WATERMARK_BITS = np.random.choice([0, 1], size=(1024,))
    NBITS = 1024
    ORDER = 32

    if len(sys.argv) != 5:
        print("Usage: uv run kraw_watermark_all.py <P1> <P2> <STRENGTH> <in_dir>")
        print("<out_dir> will be created automatically using the input directory.")
        print("P1 and P2 are the position (X, Y) parameters.")
        print("STRENGTH is the strength of the watermark.")
        print("Example: uv run kraw_watermark_all.py 0.5 0.5 100 datasets/ctscan/raw")
        print("Note: The output directory will be datasets/ctscan/krawtchouk_0.5_0.5_100")
        sys.exit(1)

    P1 = float(sys.argv[1])
    P2 = float(sys.argv[2])
    STRENGTH = int(sys.argv[3])

    in_path = sys.argv[4]

    base_query = f"{in_path}/**/"
    images = []
    images += glob.glob(base_query + "*.png", recursive=True)
    images += glob.glob(base_query + "*.jpg", recursive=True)

    good_images = []
    for image in images:
        if not os.path.isdir(image):
            good_images.append(image)

    images = good_images

    t0 = time.time()

    for i, path in enumerate(images):
        elapsed = time.time() - t0
        estimated = len(images) * (elapsed / (i+1))
        remaining = estimated-elapsed
        print(f"Processing... [{i+1}/{len(images)}] (Remaining: {remaining:.2f}s)", end="\r")
        watermark_a(path, in_path, WATERMARK_BITS, ORDER, STRENGTH, P1, P2, NBITS)