"""Central configuration for the StromNet-Erebus repository.

This is the ONLY file you should need to edit before running the notebooks.
Every notebook begins by importing this module, so a path set here is set
everywhere.

Quick start
-----------
1. Download ``Dataset_2014_2015.zip`` from the Zenodo record (see README)
   and unzip it so that the three class folders live at::

       <repo>/data/Dataset_2014_2015/Large Eruptions/
       <repo>/data/Dataset_2014_2015/Small Eruptions/
       <repo>/data/Dataset_2014_2015/No Eruptions/

2. If you are running the provenance notebooks (P1-P4), also download the
   full monochrome archive (~19 GB) and set RGB_ARCHIVE_DIR / MONO_ARCHIVE_DIR
   below to wherever you unpacked it.

3. Nothing else in this file needs to change for the reproduce track (R1-R5).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (auto-detected: the folder this file lives in)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data locations -- EDIT THESE if your data lives elsewhere
# ---------------------------------------------------------------------------
# The curated, denoised training dataset (54,071 temporal-RGB images).
# Ships in the Zenodo data record, not in the git repository.
DATASET_DIR = REPO_ROOT / "data" / "Dataset_2014_2015"

# Archive of raw monochrome frames from the Peters et al. (2014) camera,
# laid out as <root>/YYYY/MM/DD/HH/<timestamp>.jpg.  Only needed for the
# provenance notebooks (P2, P3).  ~19 GB; see the Zenodo record.
MONO_ARCHIVE_DIR = Path("/path/to/Monochrome Images")

# Archive of derived temporal-RGB frames, same YYYY/MM/DD/HH layout.
# Produced by notebook P2; consumed by P3, R4, and (optionally, for viewing
# review images) R5.
RGB_ARCHIVE_DIR = Path("/path/to/RGB Images")

# ---------------------------------------------------------------------------
# Files shipped with the repository (no edits needed)
# ---------------------------------------------------------------------------
MODEL_PATH = REPO_ROOT / "models" / "lava_lake_classifier_inference.keras"

# The seismic cross-correlation catalog: 117 verified large-eruption times
# (December 2013 - December 2014).  Stacked into the master waveform in P1
# and used to harvest training frames in P3.
MASTERLIST_PATH = REPO_ROOT / "data" / "MasterList.txt"

# Per-image softmax probabilities for every December 2015 frame, produced by
# notebook R4 from the trained model.  Provided so that the December 2015
# analysis (R5) can run without the 19 GB image archive.
CSV_2015_12 = REPO_ROOT / "data" / "StromNet_2015_12.csv"

# Manual-review outcomes for the December 2015 evaluation (one row per
# reviewed event: predicted class, human-verified true class, stratum,
# Horvitz-Thompson weight).  Ships with the repository; R5 rebuilds Table 2
# and the population-weighted F1 from it.  A fresh review made with R5's
# interactive appendix exports to OUTPUT_DIR in the same format.
REVIEW_RESULTS_CSV = REPO_ROOT / "data" / "review_results_2015_12.csv"

# ---------------------------------------------------------------------------
# Output directory (created on demand; safe to delete)
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
# Integer labels follow the order of the model's softmax output:
#   index 0 = No Eruption, index 1 = Small Eruption, index 2 = Large Eruption
CLASS_MAP = {"No Eruption": 0, "Small Eruption": 1, "Large Eruption": 2}
CLASS_NAMES = ["No Eruption", "Small Eruption", "Large Eruption"]

# Folder names inside DATASET_DIR, keyed by integer label
CLASS_FOLDERS = {0: "No Eruptions", 1: "Small Eruptions", 2: "Large Eruptions"}

# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------
IMG_HEIGHT, IMG_WIDTH = 120, 160   # model input size (images resized on load)

# ---------------------------------------------------------------------------
# Event grouping (seconds)
# ---------------------------------------------------------------------------
# Two different time scales serve two different purposes:
#
# EVENT_GAP_SEC (10 s): the *anti-leakage* scale.  Frames of one physical
#   eruption sit ~2 s apart and share source frames across the 3-channel
#   temporal composites, so anything within 10 s must stay in the same event
#   (and therefore the same train/validation/test fold).
#
# NO_ERUPTION_GROUPING_SEC (300 s): the *background* scale.  The lava lake's
#   surface cycles over 5-18 minutes, so background frames minutes apart are
#   near-identical.  Grouping background at 300 s keeps those near-duplicates
#   in the same fold.
EVENT_GAP_SEC = 10
NO_ERUPTION_GROUPING_SEC = 300

# ---------------------------------------------------------------------------
# Operating thresholds (optimized on the held-out test set; see paper §Results)
# ---------------------------------------------------------------------------
THRESH_LARGE = 0.3523   # "Large Eruption" fires only above this probability
THRESH_SMALL = 0.4208   # "Small Eruption" fires only above this probability
THRESH_NO = 0.5805      # reference value for the "No Eruption" class

# ---------------------------------------------------------------------------
# December 2015 evaluation parameters (notebook R5)
# ---------------------------------------------------------------------------
SMOOTH_LARGE = 5          # forward moving-average window for Large Eruption
SMOOTH_OTHER = 3          # forward moving-average window for Small / No
LARGE_EVENT_GAP_SEC = 1800   # 30 min: large eruptions cluster episodically
NO_ERUPTION_BUFFER_SEC = 10  # review-sampling distance from any detection
RANDOM_SEED = 42

# Stratified sampling plan for the No-Eruption review set.  Bins are on the
# "runner-up" probability (max of the Large / Small smoothed probabilities).
# target=None means review every item in the bin.
STRATA = [
    {"name": "near_threshold", "lo": 0.30, "hi": 1.01, "target": None},
    {"name": "moderate",       "lo": 0.15, "hi": 0.30, "target": 300},
    {"name": "low",            "lo": 0.05, "hi": 0.15, "target": 300},
    {"name": "clear",          "lo": 0.00, "hi": 0.05, "target": 300},
]


def seconds_since_epoch(timestamps):
    """Convert a pandas datetime Series to integer seconds since the epoch.

    Written this way because pandas >= 3.0 creates datetime64[us] columns by
    default while pandas 2.x created datetime64[ns]; a bare
    ``astype('int64') // 1_000_000_000`` silently returns wrong units on the
    microsecond dtype.  Forcing nanosecond resolution first is correct on
    both.
    """
    return timestamps.astype("datetime64[ns]").astype("int64").to_numpy() // 1_000_000_000


def describe():
    """Print the resolved configuration so notebooks can show it at startup."""
    print("StromNet-Erebus configuration")
    print(f"  Repository root:    {REPO_ROOT}")
    print(f"  Training dataset:   {DATASET_DIR}"
          + ("" if DATASET_DIR.exists() else "   [NOT FOUND - download from Zenodo]"))
    print(f"  Trained model:      {MODEL_PATH}"
          + ("" if MODEL_PATH.exists() else "   [NOT FOUND]"))
    print(f"  Dec 2015 CSV:       {CSV_2015_12}"
          + ("" if CSV_2015_12.exists() else "   [NOT FOUND]"))
    print(f"  Review results:     {REVIEW_RESULTS_CSV}"
          + ("" if REVIEW_RESULTS_CSV.exists() else "   [NOT FOUND]"))
    print(f"  Monochrome archive: {MONO_ARCHIVE_DIR}"
          + ("" if MONO_ARCHIVE_DIR.exists() else "   [provenance track only]"))
    print(f"  RGB archive:        {RGB_ARCHIVE_DIR}"
          + ("" if RGB_ARCHIVE_DIR.exists() else "   [provenance track only]"))
    print(f"  Outputs:            {OUTPUT_DIR}")
