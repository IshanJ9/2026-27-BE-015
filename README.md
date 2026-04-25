# Audio-Visual Packet Loss Concealment System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Overview
The **Audio-Visual Packet Loss Concealment System** is an advanced machine learning pipeline that performs cross-modal restoration of corrupted video and audio streams. By utilizing sophisticated pre-trained neural networks—including Wav2Vec2 and Stable Diffusion VAE—alongside signal processing techniques like Griffin-Lim and Poisson seamless cloning, the system adeptly reconstructs lost audio segments using visual cues, and restores degraded visual frames (specifically mouth regions) utilizing acoustic features.

## Features
- **Cross-Modal Restoration**:
  - **Audio to Video (Lip Recovery)**: Retrieves missing visual lip movements using intact audio chunks.
  - **Video to Audio (Speech Recovery)**: Reconstructs lost audio packets leveraging visual phoneme/mouth movements.
- **Advanced Corruption Simulation**: Injects realistic packet loss scenarios, simulating network degradation by dropping audio sequences and masking visual facial/mouth data.
- **State-of-the-Art Neural Encoding**:
  - `Wav2Vec2` for dense acoustic feature extraction.
  - `Stable Diffusion VAE` for high-fidelity latent video frame representations.
- **Robust Post-Processing**: Includes seamless cloning, LAB color space skin-tone matching, and feathered edge blending to eliminate visual artifacts during frame reconstruction.
- **Comprehensive Evaluation Suite**: Automatically generates quantitative assessments comparing the restored streams against base truth using SSIM, PSNR, STOI, and LSE-D metrics.

## Core Restoration Methodologies

### 1. Video Restoration (Audio $\rightarrow$ Video)
* **Goal**: Reconstruct corrupted or missing mouth regions using audio signals.
* **Mechanism**: 
  - **Feature Extraction**: Extracts deep acoustic embeddings using **Wav2Vec2** from the unbroken audio track.
  - **Reference Matching**: Scans a dynamic bank of uncorrupted mouth crops to find the optimal nearest-neighbor via cosine similarity matching on the audio embeddings.
  - **Seamless Compositing**: Takes the best-matching mouth crops, temporally smoothes them, and surgically pastes them back into the corrupted hole using **LAB color space skin-tone matching** and **feathered-edge blending**. (Note: No generative diffusion model is used to synthesize the mouth).

### 2. Audio Restoration (Video $\rightarrow$ Audio)
* **Goal**: Synthesize missing audio packets using the speaker's lip movements.
* **Mechanism**: 
  - **Spectral Interpolation**: Identifies gaps in the audio STFT spectrogram and interpolates the missing magnitude values using surrounding uncorrupted context.
  - **Phase Reconstruction**: Uses the **Griffin-Lim algorithm** to invert the completed magnitude spectrogram back into a time-domain audio waveform.
  - **Visual Energy Modulation**: Calculates the motion energy of the lips from the video frames (optical flow). This lip-movement envelope is multiplied against the reconstructed audio to inject natural speech prosody (volume rises and falls naturally with jaw opening/closing).

## Pipeline Architecture
The execution pipeline is modularized into seven distinct mathematical and processing stages:
1. **Data Loading**: Ingests video (`.mp4`), extracting sequential frames and sampling audio at an optimal target rate (`16kHz`).
2. **Stream Corruption**: Randomly nullifies specific audio ranges and visually masks mouth geometries to mimic true packet loss.
3. **Encoding**: Processes the isolated streams via pretrained encoders (`Wav2Vec2` & `VAE`) mapping into latent spatial and acoustic embeddings.
4. **Audio $\rightarrow$ Video Restoration**: Projects audio embeddings back into the visual domain to reconstruct and seamlessly blend the speaker's mouth.
5. **Video $\rightarrow$ Audio Restoration**: Maps facial/mouth sequences via Griffin-Lim inversions to predict and synthesize missing speech chunks.
6. **Quality Evaluation**: Computes scoring metrics to evaluate synchronization constraints and perceptual quality.
7. **Output Writing**: Renders high-quality comparative multiplexed outputs and composite `.mp4` payloads.

## Installation

### Prerequisites
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) installed and accessible in your system `PATH`.
- A discrete GPU with CUDA capability is highly recommended for model inference.

### Setup
Clone the repository and install the project dependencies:

```bash
git clone https://github.com/your-username/av-packet-loss-concealment.git
cd av-packet-loss-concealment

# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

You can launch the entire end-to-end pipeline with a single command passing your input video:

```bash
python main.py --input path/to/your/video.mp4
```

To run a quick demonstration using specific dataset configurations:
```bash
python main.py --input auto --max-frames 150 --output-dir results
```

### CLI Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | Path to the target input video (`.mp4`). Use `auto` to select from dataset. | **Required** |
| `--output-dir` | Directory to save restored `.mp4` outputs and `metrics.json`. | `outputs` |
| `--max-frames` | Limit the number of frames to process (useful for quick testing). | `150` |
| `--start-frame` | Index to commence processing (bypasses video introductions). | `500` |
| `--audio-corrupt-ratio` | Proportion of audio packets to synthetically drop. | `0.3` |
| `--mouth-expand` | Factor to expand the MediaPipe bounding box of the mouth. | `1.4` |
| `--device` | Device compute fallback: `cuda` or `cpu`. | Auto |
| `--skip-vae` | Bypasses the VAE encoding stage to save VRAM footprint. | `False` |
| `--no-comparison`| Suppresses the generation of side-by-side comparison output videos. | `False` |

## Project Structure
```text
cap/
├── 01_raw_youtube/        # Dataset repository for raw inputs.
├── models/                # Local weights and model checkpoint caching.
├── modules/
│   ├── audio_encoder.py   # Latent feature extraction via Wav2Vec2.
│   ├── audio_to_video.py  # Restores mouth frames using acoustic inputs.
│   ├── corruptor.py       # Simulates missing packets (audio & visual blanks).
│   ├── data_loader.py     # Parses input AV stream using OpenCV and Librosa.
│   ├── output_writer.py   # Assembles multiplexed comparison clips.
│   ├── sync_evaluator.py  # Generates SSIM, PSNR, STOI, and LSE-D values.
│   ├── video_encoder.py   # Compresses frames using Stable Diffusion VAE.
│   └── video_to_audio.py  # Predicts soundscapes from video frames.
├── outputs/               # Default render destination for pipeline results.
├── batch_generate.py      # Script for batch processing dataset videos.
├── main.py                # Pipeline entry point & orchestrator.
└── requirements.txt       # Python package dependencies.
```

## Performance Metrics Output
Post execution, the system saves a comprehensively structured `metrics.json` file analyzing:
- **SSIM (Structural Similarity Index)**: Contextual visual degradation check between original and restored visual.
- **PSNR (Peak Signal-to-Noise Ratio)**: Standard absolute error visual accuracy metric.
- **STOI (Short-Time Objective Intelligibility)**: Measures human-perceived audio clarity indexing against frequency variants.
- **LSE-D (Lip-Sync Error Distance)**: Determines mathematically how well constructed audio alignments match lip physical movements based on sync-net embeddings.

## Acknowledgements
Designed for high-fidelity communication restoration, leveraging the capabilities of `HuggingFace Transformers`, pre-trained `Wav2Vec2`, `Stable Diffusion / diffusers`, and `MediaPipe` face mesh extraction.
