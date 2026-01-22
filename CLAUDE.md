# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gymnasium-recorder is a Python tool for recording and replaying gameplay from Gymnasium environments (Atari, Stable-Retro, VizDoom). It captures frames and actions, stores them as Hugging Face datasets, and can replay recordings to verify environment determinism.

## Development Setup

```bash
# Create and activate conda environment
source activate-env.sh

# Set up environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

The conda environment is defined in `environment.yml` and includes Python 3.11, gymnasium, pygame, datasets, and platform-specific game engines (ale-py, vizdoom).

## Core Commands

### Recording gameplay
```bash
python main.py record <env_id>
python main.py record BreakoutNoFrameskip-v4 --fps 30
```

### Replaying recorded datasets
```bash
python main.py playback <env_id>
python main.py playback BreakoutNoFrameskip-v4 --fps 30
```

### Listing available environments
```bash
python main.py list_environments
```

## Architecture

### Single-File Design
All code is in `main.py` (~880 lines). The project prioritizes simplicity over modularization.

### Key Components

**DatasetRecorderWrapper** (main.py:210-591)
- Gymnasium wrapper that handles recording and playback
- Manages pygame rendering (2x scaled display)
- Records frames to temporary JPEG files, then converts to HF Dataset
- Handles three environment types with different action spaces:
  - Atari (ALE-py): Discrete actions
  - VizDoom: MultiBinary actions (or Dict with binary/continuous)
  - Stable-Retro: MultiBinary actions, platform-specific mappings

**Environment Creation** (main.py:674-701)
- `create_env()` detects environment type by pattern matching env_id
- Adds `_env_id`, `_vizdoom`, or `_stable_retro` attributes to distinguish backends
- Each backend has different initialization requirements and action spaces

**Action Mapping System**
- Platform-specific key bindings: `ATARI_KEY_BINDINGS`, `VIZDOOM_KEY_BINDINGS`, `STABLE_RETRO_KEY_BINDINGS`
- VizDoom requires button index mapping (main.py:304-340)
- Stable-Retro has per-console mappings (Nes, Snes, Genesis, etc.)
- Action conversion for dataset storage (main.py:277-286): normalizes numpy/dict/int to serializable format

**Dataset Management**
- Naming convention: `{username}/GymnasiumRecording__{env_id_underscored}`
- Concatenates new recordings with existing datasets on Hub
- Auto-generates dataset cards with episode/frame statistics
- Fields: episode_id, timestamp, image (HFImage), step, action

**FPS Handling** (main.py:718-755)
- Attempts to read from environment metadata first
- Falls back to defaults: Atari=15fps, VizDoom=35fps, Retro=60fps
- Pattern matches env_id when metadata unavailable

## Important Patterns

### Action Space Handling
The wrapper must handle three distinct action space types:
1. **Discrete** (Atari): Single integer action
2. **MultiBinary** (VizDoom, Retro): Array of 0s and 1s
3. **Dict** (Some VizDoom configs): `{"binary": ..., "continuous": ...}`

Action conversion happens in two places:
- Recording: Convert env actions to serializable format (main.py:277-286)
- Playback: Convert stored actions back to env format (main.py:561-580)

### Environment Detection
Use the `_vizdoom` and `_stable_retro` attributes added by `create_env()` to branch behavior:
```python
if hasattr(self.env, '_vizdoom') and self.env._vizdoom:
    # VizDoom-specific logic
elif hasattr(self.env, '_stable_retro') and self.env._stable_retro:
    # Stable-Retro-specific logic
else:
    # Atari/ALE-py logic
```

### Frame Extraction
VizDoom environments may return dict observations. Extract the actual image:
```python
if isinstance(frame, dict):
    for k in ["obs", "image", "screen"]:
        if k in frame:
            frame = frame[k]
            break
```

### User Controls
- Space: Start recording
- ESC: Exit
- Platform-specific game controls (arrow keys, Z/X buttons, etc.)

## Key Constraints

- **No testing infrastructure**: The project has no tests. Changes must be manually verified.
- **Single episode per recording session**: Each `record()` call creates one episode.
- **Pygame dependency**: All rendering and input uses pygame. The screen is created lazily after first observation.
- **Async design**: Main loop uses `asyncio` with `await asyncio.sleep()` for frame pacing.
- **Environment variables**: Requires `HF_TOKEN` in `.env` for dataset uploads.
- README.md must be kept up to date with any significant project changes using the readme-generator skill.

## Current TODOs

From TODO.md:
- Add Doom WAD support
- Add validation for dataset fields
