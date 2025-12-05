# src/extract_features.py

import os
import time
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from skimage import io, color, filters, img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops

import mlflow


# ---- Helper functions ----

def load_gray_image(path: Path):
    img = io.imread(str(path))
    # Ensure grayscale
    if img.ndim == 3:  # RGB
        img = color.rgb2gray(img)
    return img


def compute_entropy(img):
    # entropy filter expects uint8
    img_u8 = img_as_ubyte(img)
    return rank.entropy(img_u8, disk(3))


def compute_filters(img):
    """Return dict of filtered images (including original)."""
    entropy_img = compute_entropy(img)
    gaussian_img = filters.gaussian(img, sigma=1)
    sobel_img = filters.sobel(img)
    prewitt_img = filters.prewitt(img)
    hessian_img = filters.hessian(img, sigma=1)
    gabor_real, gabor_imag = filters.gabor(img, frequency=0.6)

    return {
        "orig": img,
        "entropy": entropy_img,
        "gaussian": gaussian_img,
        "sobel": sobel_img,
        "prewitt": prewitt_img,
        "hessian": hessian_img,
        "gabor_real": gabor_real,
        "gabor_imag": gabor_imag,
    }


def compute_glcm_features(gray_img, prefix):
    """
    Compute GLCM features for one image.
    gray_img should be uint8 or uint16, but we'll discretize.
    """
    img_u8 = img_as_ubyte(gray_img)

    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    props = ["contrast", "dissimilarity", "homogeneity",
             "ASM", "energy", "correlation"]

    glcm = graycomatrix(
        img_u8,
        distances=distances,
        angles=angles,
        symmetric=True,
        normed=True,
    )

    features = {}
    for p in props:
        vals = graycoprops(glcm, p)[0]  # shape (len(angles),)
        mean_val = float(np.mean(vals))
        features[f"{prefix}_glcm_{p}"] = mean_val

    return features


def summarize_basic_stats(img, prefix):
    return {
        f"{prefix}_mean": float(np.mean(img)),
        f"{prefix}_std": float(np.std(img)),
        f"{prefix}_min": float(np.min(img)),
        f"{prefix}_max": float(np.max(img)),
    }


def process_single_image(path: Path, root_yes: Path, root_no: Path):
    """
    Process one image and return (image_id, label, feature_dict)
    """
    # Image ID = relative path string
    rel_path = None
    label = None

    if root_yes in path.parents:
        rel_path = path.relative_to(root_yes.parent)  # tumor_images/yes/xxx
        label = 1
    elif root_no in path.parents:
        rel_path = path.relative_to(root_no.parent)
        label = 0
    else:
        # Fallback: treat as unknown label
        rel_path = path.name
        label = -1

    image_id = str(rel_path)

    img = load_gray_image(path)
    filt_dict = compute_filters(img)

    all_features = {}
    # For each filter result, add basic stats + GLCM
    for name, fimg in filt_dict.items():
        all_features.update(summarize_basic_stats(fimg, name))
        all_features.update(compute_glcm_features(fimg, name))

    return image_id, label, all_features


# ---- Main script ----

def main(args):
    mlflow.start_run()

    start_time = time.time()

    input_dir = Path(args.input_data)
    output_path = Path(args.output_path)

    # We assume structure:
    # tumor_images/
    #   yes/
    #   no/
    yes_dir = input_dir / "yes"
    no_dir = input_dir / "no"

    yes_images = list(yes_dir.rglob("*.png")) + list(yes_dir.rglob("*.jpg")) + list(yes_dir.rglob("*.jpeg"))
    no_images = list(no_dir.rglob("*.png")) + list(no_dir.rglob("*.jpg")) + list(no_dir.rglob("*.jpeg"))

    all_paths = yes_images + no_images
    if not all_paths:
        raise ValueError(f"No images found under {input_dir}")

    print(f"Found {len(all_paths)} images")

    # Multiprocessing
    results = []
    worker = partial(process_single_image, root_yes=yes_dir, root_no=no_dir)

    with ProcessPoolExecutor() as ex:
        future_to_path = {ex.submit(worker, p): p for p in all_paths}
        for fut in as_completed(future_to_path):
            p = future_to_path[fut]
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"Error processing {p}: {e}")

    # Build DataFrame
    rows = []
    for image_id, label, feats in results:
        row = {"image_id": image_id, "label": label}
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Log metrics
    num_images = df.shape[0]
    num_features = df.shape[1] - 2  # minus image_id, label
    extraction_time_seconds = time.time() - start_time
    compute_sku = os.environ.get("AZUREML_COMPUTE", "unknown")

    print(f"num_images: {num_images}")
    print(f"num_features: {num_features}")
    print(f"extraction_time_seconds: {extraction_time_seconds:.2f}")
    print(f"compute_sku: {compute_sku}")

    mlflow.log_metric("num_images", num_images)
    mlflow.log_metric("num_features", num_features)
    mlflow.log_metric("extraction_time_seconds", extraction_time_seconds)
    mlflow.log_param("compute_sku", compute_sku)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Parquet
    df.to_parquet(output_path, index=False)

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to tumor_images_raw folder (yes/no subfolders).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output Parquet file path (e.g. ./outputs/features.parquet).",
    )

    args = parser.parse_args()
    main(args)
