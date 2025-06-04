import re
import os
import time
import numpy as np
import pygame
import threading
import asyncio
import tempfile
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Sequence, Image as HFImage, load_dataset, concatenate_datasets
from huggingface_hub import whoami
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

import gymnasium as gym

REPO_PREFIX = "GymnasiumRecording__"

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
        self.key_to_action = {
            pygame.K_UP: 1,
            pygame.K_RIGHT: 2,
            pygame.K_LEFT: 3,
            pygame.K_DOWN: 4,
        }
        self._vizdoom_buttons = None
        self.noop_action = 0

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
        # Normalize action format for dataset storage
        if isinstance(action, np.ndarray):
            action = action.tolist()
        else:
            action = [int(action)]
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

    def _init_vizdoom_key_mapping(self):
        """Map important action names to their button indices."""
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

    def _get_user_action(self):
        """
        Map pressed keys to actions, handling Atari (Discrete), stable-retro (MultiBinary), and VizDoom (MultiBinary) environments.
        """
        with self.key_lock:
            if hasattr(self.env, '_vizdoom') and self.env._vizdoom:
                if self._vizdoom_buttons is None:
                    self._vizdoom_buttons = self._init_vizdoom_key_mapping()
                n_buttons = self.env.unwrapped.num_binary_buttons
                action = np.zeros(n_buttons, dtype=np.int32)

                pressed = self.current_keys
                alt = pygame.K_LALT in pressed or pygame.K_RALT in pressed

                def press(name):
                    idx = self._vizdoom_buttons.get(name)
                    if idx is not None and idx < n_buttons:
                        action[idx] = 1

                if pygame.K_UP in pressed:
                    press("MOVE_FORWARD")
                if pygame.K_DOWN in pressed:
                    press("MOVE_BACKWARD")

                if pygame.K_LEFT in pressed:
                    press("MOVE_LEFT" if alt else "TURN_LEFT")
                if pygame.K_RIGHT in pressed:
                    press("MOVE_RIGHT" if alt else "TURN_RIGHT")

                if pygame.K_LSHIFT in pressed or pygame.K_RSHIFT in pressed:
                    press("SPEED")
                if pygame.K_LCTRL in pressed or pygame.K_RCTRL in pressed:
                    press("ATTACK")
                if pygame.K_SPACE in pressed:
                    press("USE")

                for i in range(1, 8):
                    key_const = getattr(pygame, f"K_{i}")
                    if key_const in pressed:
                        press(f"SELECT_WEAPON{i}")

                for i, combo in enumerate(self.env.unwrapped.button_map):
                    if np.array_equal(combo, action):
                        return i
                return 0
            # ...existing code for stable-retro...
            if hasattr(self.env, '_stable_retro') and self.env._stable_retro:
                # SuperMarioBros-Nes: MultiBinary(8) action space
                action = np.zeros(self.env.action_space.n, dtype=np.int32)  # [B, Select, Start, Up, Down, Left, Right, A]
                #['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

                # @tsilva HACK: clean this up
                for key in self.current_keys:
                    if key == pygame.K_z:  # A
                        action[0] = 1
                    elif key == pygame.K_q:  # SELECT
                        action[2] = 1
                    if key == pygame.K_r:  # START
                        action[3] = 1
                    elif key == pygame.K_UP:  
                        action[4] = 1
                    elif key == pygame.K_DOWN: 
                        action[5] = 1
                    elif key == pygame.K_LEFT:   
                        action[6] = 1
                    elif key == pygame.K_RIGHT: 
                        action[7] = 1
                    if key == pygame.K_x:  # B
                        action[8] = 1

                #import numpy as np

                #action[0] = 1
                #action[7] = 1
                #action[8] = 1

                return action
            else:
                # Atari: Discrete action space
                for key in self.current_keys:
                    if key in self.key_to_action:
                        return self.key_to_action[key]
                return self.noop_action

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
            # Actions may be stored as lists in the dataset, convert back to the
            # environment's expected format
            if isinstance(action, list):
                if isinstance(self.env.action_space, gym.spaces.Discrete) and len(action) == 1:
                    action = action[0]
                else:
                    action = np.array(action, dtype=np.int32)
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
    try:
        loaded_dataset = load_dataset(hf_repo_id, split="train")
        # older versions stored actions as integers; normalize to list format
        if isinstance(loaded_dataset.features["action"], Value):
            loaded_dataset = loaded_dataset.map(
                lambda row: {"action": [row["action"]]}
            )
            loaded_dataset = loaded_dataset.cast_column(
                "action", Sequence(Value("int64"))
            )
    except Exception:
        loaded_dataset = None

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
