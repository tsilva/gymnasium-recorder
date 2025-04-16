# 🎮 gymnasium-recorder

🎬 Record and replay Atari gameplay as Hugging Face datasets with ease

## 📖 Overview

Gymnasium Recorder is a tool that wraps Gymnasium environments (specifically Atari games) to record gameplay sessions as datasets. It captures frames and actions, saving them locally or uploading them directly to Hugging Face Hub. The tool also supports replaying recorded sessions to verify environment determinism.

The recorder provides an interactive pygame interface for playing Atari games while recording your actions, making it easy to create training datasets for reinforcement learning models.

## 🚀 Installation

1. Clone the repository
2. Create and activate the conda environment:
   ```bash
   source activate-env.sh
   ```
3. Create a `.env` file with your Hugging Face token (copy from `.env.example`)

## 🛠️ Usage

### Recording a gameplay session

```bash
python main.py record BreakoutNoFrameskip-v4
```

This will open a pygame window where you can play Breakout. Use the following controls:
- Space: Action 1
- Right Arrow: Action 2
- Left Arrow: Action 3

The session will be saved locally in the `datasets` directory and uploaded to Hugging Face Hub if you've provided a token.

### Replaying a recorded session

```bash
python main.py playback username/breakoutnoframeskip_v4
```

This will replay a previously recorded session from Hugging Face Hub, verifying that the environment behaves deterministically.

## 📄 License

This project is licensed under the [MIT License](LICENSE).