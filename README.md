# StromNet-Erebus

Code, trained model, and data pipeline for:

> Dye, B. C., & Morra, G. (2026). *Detection of Small and Large Strombolian
> Eruptions in Noisy Multitemporal Infrared Images with a CNN: Toward Continuous
> Monitoring.* PNAS Nexus. DOI: []

StromNet is a lightweight convolutional neural network (597,987 parameters)
that classifies temporal infrared images of the Ray lava lake atop Mount
Erebus, Antarctica, into **no eruption**, **small eruption**, and 
**large eruption**. Each input image packs three consecutive thermal frames 
(about 2 s apart) into the red, green, and blue channels, so motion between
frames appears as color while the static crater appears gray.

**Archived releases:** code DOI (https://doi.org/10.24433/CO.7211757.v1), data DOI https://doi.org/10.15784/600381)

---

## Table of Contents

| Path | Contents |
|---|---|
| `config.py` | All paths and constants. |
| `models/lava_lake_classifier_inference.keras` | The published trained model (2.4 MB). |
| `data/MasterList.txt` | The seismic catalog: 117 verified large-eruption times (Dec 2013 to Dec 2014) behind the master waveform (P1) and the training-frame harvest (P3). |
| `data/StromNet_2015_12.csv` | Per-image model probabilities for all of December 2015 (1,164,270 rows). |
| `data/review_results_2015_12.csv` | Manual-review outcomes for the December 2015 evaluation (2,930 events). Lets R5 rebuild Table 2 and the population-weighted F1 directly. |
| `notebooks/reproduce/` | R1 to R5: everything needed to reproduce the paper's results. |
| `notebooks/provenance/` | P1 to P4: how the shipped data was originally built (needs the raw archive). |
| `PARITY_REPORT.md` | Verification that this repository reproduces the paper's numbers. |
| `CHANGES.md` | Every fix applied relative to the original research notebooks. |

Not in this repository (too large for git; download from the Zenodo data
record and see `data/README.md`):

- `Dataset_2014_2015.zip` (about 350 MB): the curated training dataset
  (54,071 multitemporal-RGB images in three class folders).
- The monochrome frame archive (about 19 GB): raw camera frames, December
  2013 to January 2016, needed only for the provenance track.

## Setup

Requirements: Python 3.10 to 3.13, about 1 GB of disk for the repository plus the unzipped
training dataset, and an internet connection for the install. A
CUDA-capable GPU is needed only for training (R2 and P4); everything else
runs on CPU. `requirements.txt` pins the verified versions, including
Keras 3.13.2, the exact version that saved the shipped model. `requirements.txt` defaults to CPU only, but contains the line for GPU if the user has a CUDA-capable GPU.

### Ubuntu (24.04/26.04)
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip unzip
cd stromnet-erebus
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training on an NVIDIA card, install the CUDA-enabled build instead
of the plain one (NVIDIA driver 525 or newer required):

```bash
pip install "tensorflow[and-cuda]==2.21.*"
```

### Windows 10/11

1. Install Python 3.12 from python.org. In the installer, check
   "Add python.exe to PATH".
2. In PowerShell:

```powershell
cd stromnet-erebus
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, either run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command
Prompt with `.venv\Scripts\activate.bat` instead.

TensorFlow on native Windows is CPU-only. That covers every notebook except
the training ones; to train (R2, P4) on a Windows machine with an NVIDIA
GPU, install WSL2 with Ubuntu and follow the Ubuntu instructions inside it.

### macOS (Apple silicon)

Requires an Apple-silicon Mac (M1 or later); TensorFlow 2.21 does not
publish Intel-Mac builds. Install Python 3.12 from python.org or with
Homebrew (`brew install python@3.12`), then in Terminal:

```bash
cd stromnet-erebus
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, for GPU acceleration on the Apple GPU:
`pip install tensorflow-metal`.

### Get the dataset

Download `Dataset_2014_2015.zip` from the Zenodo data record and unzip it
so the three class folders sit at `data/Dataset_2014_2015/` inside the
repository (or unzip anywhere and point `config.DATASET_DIR` at it). The
December 2015 analysis (R5) does not need it; the dataset is required for
R1, R2, R3, and P4.

## Launching the notebooks

From the repository root, with the virtual environment active:

```bash
jupyter lab
```

(Windows: run the same command in the PowerShell window where you
activated `.venv`.) A browser tab opens with the JupyterLab file browser.
Open `notebooks/reproduce/`, double-click a notebook, and run it top to
bottom: either press Shift+Enter cell by cell, or use the menu
Run > Run All Cells. Each notebook prints the numbers you should expect
alongside the ones it computes.

When you are done, close the browser tab, press Ctrl+C in the terminal to
stop JupyterLab, and type `deactivate` to leave the virtual environment.
Next session, only two commands are needed again: activate the environment,
then `jupyter lab`.

## Which notebook do I run?

Three tracks. Each notebook is self-contained, runs top to bottom, and
states its expected outputs.

**Track A: analyze without training (CPU, about 30 min).**
1. `R1_verify_dataset`: integrity-check the downloaded dataset (skippable
   if you only want December 2015).
2. `R5_december2015_analysis`: the temporal-generalization result. It
   reproduces the 49 large and 420 small detections, the stratified review
   design, Table 2, the population-weighted F1, and the Figure 6 weekly
   panels, entirely from the shipped CSV files.

**Track B: retrain from scratch (GPU, roughly overnight for the full 10
runs).**
1. `R1_verify_dataset`
2. `R2_train_model`: 10 independent training runs, variance statistics,
   winner selection.
3. `R3_evaluate_model`: held-out-test metrics, confusion matrix, per-class
   threshold optimization, ROC and calibration.
4. `R4_classify_new_month`: run the model over a month of imagery
   (requires the RGB archive).

**Track C: provenance (requires the raw 19 GB archive and/or IRIS access;
multi-day GPU compute for P4).** P1 seismic cross-correlation, then P2
monochrome-to-RGB conversion, then P3 clean-dataset assembly, then P4
Confident Learning denoising. Every human-in-the-loop step is marked
"Manual step (output provided)": the shipped dataset is the final
product of this track, so these notebooks are documentation, not
prerequisites.

## Licenses

- **Code** (notebooks, `config.py`): MIT; see `LICENSE`.
- **Data and model weights** (`data/`, `models/`, and the Zenodo data
  record): Creative Commons Attribution 4.0 (CC BY 4.0); see
  `DATA_LICENSE`.

## Citation

See `CITATION.cff` (GitHub renders a "Cite this repository" button from
it). Please cite both the article and the archived code/data records.

## Acknowledgments

Infrared imagery courtesy of the Mount Erebus Volcano Observatory (MEVO)
and the autonomous camera system of Peters et al. (2014). Seismic data via
IRIS Data Services (ER network, CON station).
