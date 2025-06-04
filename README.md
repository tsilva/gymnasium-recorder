# 🎮 gymnasium-recorder

<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>


🎬 Record [Gymnasium ALE](https://ale.farama.org/environments/) gameplays as Hugging Face datasets with ease

## 📖 Overview

gymnasium-recorder is a tool that wraps [Gymnasium ALE environments](https://ale.farama.org/environments/) (specifically Atari games) to record gameplay sessions as datasets. It captures frames and actions during interactive play, saving them as structured datasets that can be uploaded to Hugging Face Hub.

The tool provides a pygame interface for playing Atari games while recording your actions, making it easy to create training datasets for reinforcement learning models. It also supports replaying recorded sessions to verify environment determinism.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gymnasium-recorder.git
cd gymnasium-recorder

# Create and activate the conda environment
source activate-env.sh

# Create a .env file with your Hugging Face token
cp .env.example .env
# Edit .env with your token
```

## 🛠️ Usage

### Recording a gameplay session

```bash
python main.py record BreakoutNoFrameskip-v4
```

This will open a pygame window where you can play Breakout. Use the following controls:
- Space: Action 1
- Right Arrow: Action 2
- Left Arrow: Action 3

For VizDoom environments the keys are mapped automatically so that:
- Arrow keys move forward/backward and turn left/right
- Alt + Left/Right Arrow strafes left/right
- Shift makes the agent run
- Ctrl fires the current weapon
- Spacebar performs the `USE` action (open doors, switches)
- Number keys 1–7 select weapons if available

Key mappings for Atari, VizDoom and stable-retro environments are now
configured through a shared data structure in `main.py`. Edit the mapping
there if you want to customize controls for a specific game.  Atari
mappings use action names (e.g. `RIGHT`, `FIRE`) which are converted to
the correct action indices at runtime, so unsupported actions simply
fall back to `NOOP`.

You can hold multiple direction keys together to combine actions (for example,
Up + Right to move forward while turning). Running, strafing and firing can also
be combined with movement.

The session will be automatically saved as a dataset and uploaded to Hugging Face Hub if you've provided a token.

### Replaying a recorded session

```bash
python main.py playback BreakoutNoFrameskip-v4
```

This will replay a previously recorded session from Hugging Face Hub, allowing you to verify the environment's deterministic behavior.

## 📄 License

This project is licensed under the [MIT License](LICENSE).