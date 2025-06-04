import re
import os
import time
import numpy as np
import pygame
import threading
import asyncio
import tempfile
from typing import Dict, Iterable, Optional
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Sequence, Image as HFImage, load_dataset, concatenate_datasets
from huggingface_hub import whoami
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

import gymnasium as gym

REPO_PREFIX = "GymnasiumRecording__"

# ---------------------------------------------------------------------------
# Key mapping utilities
# ---------------------------------------------------------------------------

ATARI_KEY_CONFIG = {
    "type": "discrete",
    "noop": 0,
    "mapping": {
        # Map keys to action names; indices are resolved at runtime
        pygame.K_SPACE: "FIRE",
        pygame.K_UP: "UP",
        pygame.K_RIGHT: "RIGHT",
        pygame.K_LEFT: "LEFT",
        pygame.K_DOWN: "DOWN",   # Fallback to NOOP if unsupported
    },
    "combos": {
        # Direction + fire/jump combinations
        (pygame.K_RIGHT, pygame.K_SPACE): "RIGHTFIRE",
        (pygame.K_LEFT, pygame.K_SPACE): "LEFTFIRE",
        (pygame.K_UP, pygame.K_SPACE): "UPFIRE",
        (pygame.K_DOWN, pygame.K_SPACE): "DOWNFIRE",
        # Diagonal movement
        (pygame.K_UP, pygame.K_RIGHT): "UPRIGHT",
        (pygame.K_UP, pygame.K_LEFT): "UPLEFT",
        (pygame.K_DOWN, pygame.K_RIGHT): "DOWNRIGHT",
        (pygame.K_DOWN, pygame.K_LEFT): "DOWNLEFT",
        # Diagonal + fire
        (pygame.K_UP, pygame.K_RIGHT, pygame.K_SPACE): "UPRIGHTFIRE",
        (pygame.K_UP, pygame.K_LEFT, pygame.K_SPACE): "UPLEFTFIRE",
        (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_SPACE): "DOWNRIGHTFIRE",
        (pygame.K_DOWN, pygame.K_LEFT, pygame.K_SPACE): "DOWNLEFTFIRE",
    },
}

STABLE_RETRO_KEY_CONFIG = {
    "type": "multibinary",
    # Size will be determined from the env.action_space
    "size": lambda env: env.action_space.n,
    "mapping": {
        pygame.K_z: 0,        # A
        pygame.K_q: 2,        # SELECT
        pygame.K_r: 3,        # START
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,        # B
    },
}

VIZDOOM_KEY_CONFIG = {
    "type": "vizdoom",
    "mapping": {
        pygame.K_UP: "MOVE_FORWARD",
        pygame.K_DOWN: "MOVE_BACKWARD",
        pygame.K_LSHIFT: "SPEED",
        pygame.K_RSHIFT: "SPEED",
        pygame.K_LCTRL: "ATTACK",
        pygame.K_RCTRL: "ATTACK",
        pygame.K_SPACE: "USE",
        pygame.K_1: "SELECT_WEAPON1",
        pygame.K_2: "SELECT_WEAPON2",
        pygame.K_3: "SELECT_WEAPON3",
        pygame.K_4: "SELECT_WEAPON4",
        pygame.K_5: "SELECT_WEAPON5",
        pygame.K_6: "SELECT_WEAPON6",
        pygame.K_7: "SELECT_WEAPON7",
    },
}


class KeyMapper:
    """Helper class to map pressed keys to environment actions."""

    def __init__(self, env, config: Optional[Dict] = None):
        self.env = env
        self.config = config or self._default_config()
        self._vizdoom_buttons: Optional[Dict[str, int]] = None
        self._discrete_mapping: Optional[Dict[int, int]] = None
        self._discrete_combo_mapping: Optional[Dict[frozenset, int]] = None

    # -------------------------------------------------------
    # Config helpers
    # -------------------------------------------------------
    def _default_config(self) -> Dict:
        if getattr(self.env, "_vizdoom", False):
            return VIZDOOM_KEY_CONFIG
        if getattr(self.env, "_stable_retro", False):
            return STABLE_RETRO_KEY_CONFIG
        return ATARI_KEY_CONFIG

    # -------------------------------------------------------
    # Discrete environments (Atari)
    # -------------------------------------------------------
    def _build_discrete_mapping(self):
        self._discrete_mapping = {}
        self._discrete_combo_mapping = {}
        action_meanings = []
        if hasattr(self.env.unwrapped, "get_action_meanings"):
            try:
                action_meanings = self.env.unwrapped.get_action_meanings()
            except Exception:
                action_meanings = []
        for key, mapping in self.config["mapping"].items():
            if isinstance(mapping, int):
                idx = mapping
            else:
                idx = action_meanings.index(mapping) if mapping in action_meanings else None
            if idx is not None:
                self._discrete_mapping[key] = idx
        for keys, mapping in self.config.get("combos", {}).items():
            if isinstance(mapping, int):
                idx = mapping
            else:
                idx = action_meanings.index(mapping) if mapping in action_meanings else None
            if idx is not None:
                self._discrete_combo_mapping[frozenset(keys)] = idx

    def _discrete_action(self, pressed: Iterable[int]):
        if self._discrete_mapping is None:
            self._build_discrete_mapping()
        pressed_set = set(pressed)
        # Check combinations first (longer combos have priority)
        for keys, idx in sorted(self._discrete_combo_mapping.items(), key=lambda i: len(i[0]), reverse=True):
            if keys.issubset(pressed_set):
                return idx
        for key in pressed_set:
            if key in self._discrete_mapping:
                return self._discrete_mapping[key]
        return self.config.get("noop", 0)

    # -------------------------------------------------------
    # MultiBinary environments (stable-retro)
    # -------------------------------------------------------
    def _multibinary_action(self, pressed: Iterable[int]):
        size = self.config["size"](self.env) if callable(self.config["size"]) else self.config["size"]
        action = np.zeros(size, dtype=np.int32)
        for key in pressed:
            idx = self.config["mapping"].get(key)
            if idx is not None and idx < size:
                action[idx] = 1
        return action

    # -------------------------------------------------------
    # VizDoom environments
    # -------------------------------------------------------
    def _init_vizdoom_key_mapping(self):
        available = [b.name for b in self.env.unwrapped.game.get_available_buttons()]

        def idx(name):
            return available.index(name) if name in available else None

        mapping = {
            "ATTACK": idx("ATTACK"),
            "USE": idx("USE"),
            "MOVE_LEFT": idx("MOVE_LEFT"),
            "MOVE_RIGHT": idx("MOVE_RIGHT"),
            "MOVE_FORWARD": idx("MOVE_FORWARD"),
            "MOVE_BACKWARD": idx("MOVE_BACKWARD"),
            "TURN_LEFT": idx("TURN_LEFT"),
            "TURN_RIGHT": idx("TURN_RIGHT"),
            "SPEED": idx("SPEED"),
        }

        for i in range(1, 8):
            mapping[f"SELECT_WEAPON{i}"] = idx(f"SELECT_WEAPON{i}")

        return {k: v for k, v in mapping.items() if v is not None}

    def _vizdoom_action(self, pressed: Iterable[int]):
        if self._vizdoom_buttons is None:
            self._vizdoom_buttons = self._init_vizdoom_key_mapping()
        n_buttons = self.env.unwrapped.num_binary_buttons
        action = np.zeros(n_buttons, dtype=np.int32)

        alt = pygame.K_LALT in pressed or pygame.K_RALT in pressed

        def press(name: str):
            idx = self._vizdoom_buttons.get(name)
            if idx is not None and idx < n_buttons:
                action[idx] = 1

        for key in pressed:
            if key == pygame.K_LEFT:
                press("MOVE_LEFT" if alt else "TURN_LEFT")
            elif key == pygame.K_RIGHT:
                press("MOVE_RIGHT" if alt else "TURN_RIGHT")
            else:
                name = self.config["mapping"].get(key)
                if name is None:
                    continue
                names = [name] if isinstance(name, str) else name
                for n in names:
                    press(n)

        for i, combo in enumerate(self.env.unwrapped.button_map):
            if np.array_equal(combo, action):
                return i
        return 0

    # -------------------------------------------------------
    def get_action(self, pressed: Iterable[int]):
        typ = self.config["type"]
        if typ == "discrete":
            return self._discrete_action(pressed)
        if typ == "multibinary":
            return self._multibinary_action(pressed)
        if typ == "vizdoom":
            return self._vizdoom_action(pressed)
        raise ValueError(f"Unknown key mapping type: {typ}")

class DatasetRecorderWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for recording and replaying Atari gameplay as Hugging Face datasets.
    """
    def __init__(self, env):
        super().__init__(env)

        self.recording = False
        self.frame_shape = None  # Delay initialization
        self.screen = None       # Delay initialization

        pygame.init()
        # pygame.display.set_caption will be set after env_id is available

        self.current_keys = set()
        self.key_lock = threading.Lock()
        self.key_mapper = KeyMapper(env)

        self.episode_ids = []
        self.frames = []
        self.actions = []
        self.steps = []

        self.temp_dir = tempfile.mkdtemp()

    def _ensure_screen(self, frame):
        """
        Ensure pygame screen is initialized with the correct shape.
        """
        # If frame is a dict (e.g., VizDoom), extract the image
        if isinstance(frame, dict):
            # Try common keys for image observation
            for k in ["obs", "image", "screen"]:
                if k in frame:
                    frame = frame[k]
                    break
        if self.screen is None or self.frame_shape is None:
            self.frame_shape = frame.shape
            self.screen = pygame.display.set_mode((self.frame_shape[1] * 2, self.frame_shape[0] * 2))
            pygame.display.set_caption(getattr(self.env, "_env_id", "Gymnasium Recorder"))

    def _record_frame(self, episode_id, step, frame, action):
        """
        Save a frame and action to temporary storage.
        """
        if not self.recording: return

        # If frame is a dict (e.g., VizDoom), extract the image
        if isinstance(frame, dict):
            for k in ["obs", "image", "screen"]:
                if k in frame:
                    frame = frame[k]
                    break

        img = PILImage.fromarray(frame.astype(np.uint8))
        path = os.path.join(self.temp_dir, f"frame_{len(self.frames):05d}.png")
        img.save(path, format="PNG")
        self.episode_ids.append(episode_id)
        self.steps.append(step)
        self.frames.append(path)
        #if type(action) is np.ndarray: action = int(''.join(map(str, action)), 2)
        self.actions.append(action)

    def _input_loop(self):
        """
        Handle pygame input events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            elif event.type == pygame.KEYDOWN:
                with self.key_lock: self.current_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                with self.key_lock: self.current_keys.discard(event.key)
        return True


    def _get_user_action(self):
        """
        Map pressed keys to actions, handling Atari (Discrete), stable-retro (MultiBinary), and VizDoom (MultiBinary) environments.
        """
        with self.key_lock:
            pressed = set(self.current_keys)
        return self.key_mapper.get_action(pressed)

    def _render_frame(self, frame):
        """
        Render a frame using pygame, scaled to 2x.
        """
        # If frame is a dict (e.g., VizDoom), extract the image
        if isinstance(frame, dict):
            for k in ["obs", "image", "screen"]:
                if k in frame:
                    frame = frame[k]
                    break
        self._ensure_screen(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))

        # Scale the surface to 2x the original size
        scaled_surface = pygame.transform.scale2x(surface)

        # Update display with scaled frame
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    async def record(self, fps=30):
        self.recording = True
        try: return await self._record(fps=fps)
        finally: self.recording = False

    async def _record(self, fps=30):
        self.recording = True
        try: 
            await self.play(fps=fps)
            features = Features({"episode_id": Value("int64"), "image": HFImage(), "step": Value("int64"), "action": Sequence(Value("int64"))})
            data = {"episode_id" : self.episode_ids, "image": self.frames, "step" : self.steps, "action": self.actions}
            dataset = Dataset.from_dict(data, features=features)
            return dataset
        finally: 
            self.recording = False

    async def play(self, fps=30):
        try: await self._play(fps)
        finally: self.close()

    async def _play(self, fps=30):
        """
        Main loop for interactive gameplay and recording.
        """
        clock = pygame.time.Clock()
        
        self.episode_ids.clear()
        self.frames.clear()
        self.actions.clear()
        self.steps.clear()

        episode_id = int(time.time())
        obs, _ = self.env.reset()
        self._ensure_screen(obs)  # Ensure pygame window is created after first obs
        self._render_frame(obs)
        done = False
        step = 0
        while not done:
            if not self._input_loop(): break
            action = self._get_user_action()
            self._record_frame(episode_id, step, obs, action)
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self._render_frame(obs)
            clock.tick(fps)
            await asyncio.sleep(1.0 / fps)
            step += 1 

    async def replay(self, actions, fps=30):
        clock = pygame.time.Clock()
        obs, _ = self.env.reset()
        self._ensure_screen(obs)
        self._render_frame(obs)
        for action in tqdm(actions):
            obs, _, _, _, _ = self.env.step(action)
            self._render_frame(obs)
            clock.tick(fps)
            await asyncio.sleep(1.0 / fps)

    def close(self):
        """
        Clean up resources and save dataset if needed.
        """
        pygame.quit()
        super().close()

def env_id_to_hf_repo_id(env_id):
    user_info = whoami()
    username = user_info.get("name") or user_info.get("user") or user_info.get("username")
    env_id_underscored = env_id.replace("-", "_").replace("/", "_")
    hf_repo_id = f"{username}/{REPO_PREFIX}{env_id_underscored}"
    return hf_repo_id

def create_env(env_id):
    # In case ROM is suffixed with platform name then use stable-retro
    retro_platforms = {"Nes", "GameBoy", "Snes", "Atari2600", "Genesis"}
    match = re.search(r"-(" + "|".join(retro_platforms) + r")$", env_id)
    if match:
        import retro
        #platform = match.group(1)
        #game_name = env_id.replace(f"-{platform}", "")
        env = retro.make(env_id, render_mode="rgb_array")
        env._stable_retro = True
    elif "Vizdoom" in env_id:
        from vizdoom import gymnasium_wrapper
        # allow pressing multiple buttons at once for smoother control
        env = gym.make(env_id, render_mode="rgb_array", max_buttons_pressed=4)
        env._vizdoom = True
    # Otherwise use ale_py
    else:
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make(env_id, render_mode="rgb_array")
    
    env._env_id = env_id.replace("-", "_")
    return env

def get_default_fps(env_id):
    """
    Return a sensible default FPS for the given environment.
    """
    # NES and most retro consoles: 60 FPS
    retro_platforms_60fps = {"Nes", "GameBoy", "Snes", "Genesis"}
    # Atari 2600: 60 FPS (but Gymnasium ALE often uses 15 or 30 for human play)
    if any(env_id.endswith(f"-{plat}") for plat in retro_platforms_60fps):
        return 60
    if "Atari2600" in env_id:
        return 60
    # Gymnasium ALE Atari: 15 FPS is common for human play
    if "NoFrameskip" in env_id or "ALE" in env_id or env_id.lower() in [
        "breakout", "pong", "spaceinvaders", "seaquest", "qbert", "ms_pacman"
    ]:
        return 15
    # VizDoom: 35 FPS is the engine's default
    if "Vizdoom" in env_id or "vizdoom" in env_id:
        return 35
    # Default fallback
    return 15

async def main():
    parser = argparse.ArgumentParser(description="Atari Gymnasium Recorder/Playback")
    parser.add_argument("mode", type=str, choices=["record", "playback"], help="Mode of operation: 'record' or 'playback'")
    parser.add_argument("env_id", type=str, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for playback/recording")
    args = parser.parse_args()

    env_id = args.env_id
    hf_repo_id = env_id_to_hf_repo_id(env_id)
    try: loaded_dataset = load_dataset(hf_repo_id, split="train")
    except: loaded_dataset = None

    # Determine FPS: use user value if set, otherwise use default for env
    fps = args.fps if args.fps is not None else get_default_fps(env_id)

    if args.mode == "record":
        env = create_env(env_id)
        recorder = DatasetRecorderWrapper(env)
        recorded_dataset = await recorder.record(fps=fps)
        final_dataset = concatenate_datasets([loaded_dataset, recorded_dataset]) if loaded_dataset else recorded_dataset
        final_dataset.push_to_hub(hf_repo_id)
    elif args.mode == "playback":
        assert loaded_dataset is not None, f"Dataset not found: {hf_repo_id}"
        env = create_env(env_id)
        recorder = DatasetRecorderWrapper(env)
        actions = loaded_dataset["action"]
        await recorder.replay(actions, fps=fps)

if __name__ == "__main__":
    asyncio.run(main())
