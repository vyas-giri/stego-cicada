from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input_imgs")
OUTPUT_DIR = os.path.join(DATA_DIR, "output_imgs")

MAGIC = b"STG1"