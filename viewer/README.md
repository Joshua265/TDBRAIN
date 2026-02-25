# EEG .npy Viewer

A modular PyQt6-based viewer for EEG recordings stored in `.npy` format (TDBRAIN-style).

## Features

- **Multi-view interface**: Switch between **Time series** and **PSD** (Power Spectral Density) views.
- **Interactive Time Series**:
  - Panning and zooming via mouse navigation.
  - Per-channel scaling and robust normalization.
  - Artifact overlays with detailed inspector (click to open).
- **PSD Analysis**:
  - Live PSD calculation for the visible time window.
  - Global vs. local (visible) spectra comparison.
- **Metadata Inspection**:
  - View subject info from `participants.tsv`.
  - Detailed file metadata and artifact channel summaries.
- **Glossaries**: Built-in reference for artifact types and channel naming conventions.

## Installation

### Using Nix (Recommended)

If you have Nix installed with Flakes enabled:

```bash
# Enter the development shell
nix develop

# Run the viewer
python -m eeg_viewer
```

### Manual Installation (non-Nix)

Requirements: Python 3.10+ (tested with 3.13)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the viewer**:
   ```bash
   python -m eeg_viewer
   ```

## Usage

You can run the viewer without arguments to start scanning the current directory, or provide specific paths:

```bash
# Scan a specific directory for recordings
python -m eeg_viewer --root /path/to/data

# Load a specific recording and link participants.tsv
python -m eeg_viewer --file data/sub-01_EC.npy --participants-tsv data/participants.tsv
```

### Command Line Arguments

- `--root`: Dataset root to recursively scan for `.npy` recordings.
- `--participants-tsv`: Path to `participants.tsv` (used to display subject metadata).
- `--file`: Directly open a specific `.npy` recording.

### Shared Controls

- **Mouse Navigation**:
  - **Left-drag**: Pan (horizontal only in time series).
  - **Right-drag / Scroll**: Zoom.
- **Artifacts**: Colored regions represent detected artifacts. Click on them to view details about the detector and automated scoring.
- **Gain/Spacing**: Adjust the visual height and vertical separation of EEG channels.

## File Format

The viewer expects `.npy` files containing a dictionary (or a NumPy object array containing a dictionary) with the following keys:
- `data`: (N_channels, N_samples) array.
- `Fs`: Sampling frequency.
- `labels`: List of channel names.
- `artifacts`: (Optional) Metadata about detected artifacts.
