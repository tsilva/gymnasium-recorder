<div align="center">
  <img src="logo.jpg" alt="gymnasium-recorder" width="256"/>

  # gymnasium-recorder

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg)](https://huggingface.co/docs/datasets)

  **🎮 Record and replay gameplay from Gymnasium environments as Hugging Face datasets 📊**

  [Features](#features) · [Quick Start](#quick-start) · [Usage](#usage) · [Supported Environments](#supported-environments)
</div>

---

## Features

- **Multi-platform support** - Works with Atari (ALE-py), Stable-Retro, and VizDoom environments
- **Dataset-first design** - Captures frames and actions directly as Hugging Face datasets
- **Automatic key bindings** - Platform-specific controls preconfigured for each environment type
- **Playback verification** - Replay recordings to confirm environment determinism
- **Hub integration** - Push datasets directly to Hugging Face Hub with auto-generated dataset cards

## Quick Start

```bash
# Clone the repository
git clone https://github.com/tsilva/gymnasium-recorder.git
cd gymnasium-recorder

# Install dependencies
uv sync

# Configure Hugging Face token
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

## Usage

### Record gameplay

```bash
uv run python main.py record BreakoutNoFrameskip-v4
uv run python main.py record VizdoomBasic-v0 --fps 35
uv run python main.py record Airstriker-Genesis --fps 60
```

Press **Space** to start recording. Use **ESC** to stop and exit.

### Replay a dataset

```bash
uv run python main.py playback BreakoutNoFrameskip-v4
```

Replays the recorded actions from your Hugging Face Hub dataset.

### List available environments

```bash
uv run python main.py list_environments
```

Shows all available Atari, Stable-Retro, and VizDoom environments.

## Supported Environments

| Platform | Examples | Default FPS |
|----------|----------|-------------|
| Atari (ALE-py) | `BreakoutNoFrameskip-v4`, `PongNoFrameskip-v4` | 15 |
| VizDoom | `VizdoomBasic-v0`, `VizdoomCorridor-v0` | 35 |
| Stable-Retro | `Airstriker-Genesis`, `SuperMarioBros-Nes` | 60 |

### Controls

| Platform | Controls |
|----------|----------|
| Atari | Arrow keys for movement |
| VizDoom | Arrows (move/turn), Ctrl (attack), Space (use), 1-7 (weapons) |
| Stable-Retro | Arrows, Z/X (A/B buttons), Tab/Enter (Select/Start) |

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for dependency management)
- Hugging Face account and token (for dataset uploads)

## How It Works

1. **Recording**: The `DatasetRecorderWrapper` captures each frame as a JPEG and logs the corresponding action
2. **Storage**: Frames and actions are assembled into a Hugging Face Dataset with columns: `episode_id`, `timestamp`, `image`, `step`, `action`
3. **Upload**: Datasets are pushed to Hub with naming convention `{username}/GymnasiumRecording__{env_id}`
4. **Playback**: Recorded actions are fed back to the environment to verify deterministic replay

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
