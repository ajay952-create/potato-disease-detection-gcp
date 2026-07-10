import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from training.model import build_model
from training.model import build_model

model = build_model()

model.summary()