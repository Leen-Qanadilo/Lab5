
import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_images", type=str, required=True)
    parser.add_argument("--features_output", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.features_output, exist_ok=True)
    out_file = Path(args.features_output) / "features_dummy.txt"
    with open(out_file, "w") as f:
        f.write(f"Read images from: {args.raw_images}\n")
        f.write("TODO: implement real feature extraction.\n")
    print("Feature extraction placeholder finished.")

if __name__ == "__main__":
    main()
