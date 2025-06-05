# 🎮 gymnasium-recorder

<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>

Capture and share your favorite Gymnasium moments.

## ✨ Features

- Works with Atari, Stable-Retro and VizDoom environments
- Captures frames and actions as a Hugging Face dataset
- Automatic key bindings for each platform
- Playback mode to confirm environment determinism
- Utility command to list available environments

## ⚙️ Setup

```bash
# clone
git clone https://github.com/yourusername/gymnasium-recorder.git
cd gymnasium-recorder

# install dependencies and activate the conda env
source activate-env.sh

# add your Hugging Face token
cp .env.example .env
# edit .env and fill in HF_TOKEN
```

## 🚀 Usage

### Record gameplay

```bash
python main.py record BreakoutNoFrameskip-v4
```

### Replay a dataset

```bash
python main.py playback BreakoutNoFrameskip-v4
```

### List supported environments

```bash
python main.py list_environments
```

Recorded datasets can be pushed to the hub right from the CLI.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
