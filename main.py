import re
import os
import time
import numpy as np
import pygame
import threading
import asyncio
import tempfile
import cv2
from huggingface_hub import whoami, DatasetCard, DatasetCardData, HfApi
from datasets import Dataset, Value, Sequence, Image as HFImage, load_dataset, concatenate_datasets
from huggingface_hub import whoami, DatasetCard, DatasetCardData
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv(override=True)  # Load environment variables from .env file

import gymnasium as gym

REPO_PREFIX = "GymnasiumRecording__"
START_KEY = pygame.K_SPACE

# Default key mappings for each supported environment type
ATARI_KEY_BINDINGS = {
    pygame.K_UP: 1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT: 3,
    pygame.K_DOWN: 4,
}

VIZDOOM_KEY_BINDINGS = {
    pygame.K_UP: "MOVE_FORWARD",
    pygame.K_DOWN: "MOVE_BACKWARD",
    pygame.K_LEFT: "TURN_LEFT",
    pygame.K_RIGHT: "TURN_RIGHT",
    pygame.K_LSHIFT: "SPEED",
    pygame.K_RSHIFT: "SPEED",
    pygame.K_LCTRL: "ATTACK",
    pygame.K_RCTRL: "ATTACK",
    pygame.K_SPACE: "USE",
}
for i in range(1, 8):
    VIZDOOM_KEY_BINDINGS[getattr(pygame, f"K_{i}")] = f"SELECT_WEAPON{i}"

STABLE_RETRO_KEY_BINDINGS = {
    "Nes": {
        pygame.K_z: 0,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,
    },
    "Atari2600": {
        pygame.K_z: 0,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
    },
    "Snes": {
        pygame.K_z: 0,
        pygame.K_a: 1,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,
        pygame.K_s: 9,
        pygame.K_q: 10,
        pygame.K_w: 11,
    },
    "GbAdvance": {
        pygame.K_z: 0,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,
        pygame.K_a: 10,
        pygame.K_s: 11,
    },
    "GameBoy": {
        pygame.K_z: 0,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,
    },
    "GbColor": {
        pygame.K_z: 0,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,
    },
    "PCEngine": {
        pygame.K_x: 0,
        pygame.K_c: 1,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_z: 8,
        pygame.K_a: 9,
        pygame.K_s: 10,
        pygame.K_d: 11,
    },
    "Saturn": {
        pygame.K_x: 0,
        pygame.K_z: 1,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_c: 8,
        pygame.K_a: 9,
        pygame.K_s: 10,
        pygame.K_d: 11,
    },
    "32x": {
        pygame.K_x: 0,
        pygame.K_z: 1,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_c: 8,
        pygame.K_a: 9,
        pygame.K_s: 10,
        pygame.K_d: 11,
    },
    "Genesis": {
        pygame.K_x: 0,
        pygame.K_z: 1,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_c: 8,
        pygame.K_a: 9,
        pygame.K_s: 10,
        pygame.K_d: 11,
    },
    "Sms": {
        pygame.K_z: 0,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,
    },
    "GameGear": {
        pygame.K_z: 0,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_x: 8,
    },
    "SCD": {
        pygame.K_x: 0,
        pygame.K_z: 1,
        pygame.K_TAB: 2,
        pygame.K_RETURN: 3,
        pygame.K_UP: 4,
        pygame.K_DOWN: 5,
        pygame.K_LEFT: 6,
        pygame.K_RIGHT: 7,
        pygame.K_c: 8,
        pygame.K_a: 9,
        pygame.K_s: 10,
        pygame.K_d: 11,
    },
}

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
        self.key_to_action = ATARI_KEY_BINDINGS
        self._vizdoom_buttons = None
        self._vizdoom_vector_map = None
        self.noop_action = 0

        self.episode_ids = []
        self.frames = []
        self.actions = []
        self.steps = []
        self.timestamps = []

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

        frame_uint8 = frame.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        path = os.path.join(self.temp_dir, f"frame_{len(self.frames):05d}.jpg")
        cv2.imwrite(path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self.episode_ids.append(episode_id)
        self.steps.append(step)
        self.frames.append(path)
        self.timestamps.append(time.time())
        # Normalize action format for dataset storage
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, dict):
            action = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in action.items()
            }
        else:
            action = [int(action)]
        self.actions.append(action)

    def _input_loop(self):
        """
        Handle pygame input events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                with self.key_lock:
                    self.current_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                with self.key_lock: self.current_keys.discard(event.key)
        return True

    def _init_vizdoom_key_mapping(self):
        """Map important action names to their button indices."""
        available = [b.name for b in self.env.unwrapped.game.get_available_buttons()]
        offset = self.env.unwrapped.num_delta_buttons

        def idx(name):
            if name in available:
                return available.index(name) - offset
            return None

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

        # Precompute vector -> discrete action mapping for faster lookup
        space = self.env.action_space
        if isinstance(space, gym.spaces.Dict):
            space = space.get("binary")
        if isinstance(space, gym.spaces.Discrete):
            self._vizdoom_vector_map = {
                tuple(combo): i for i, combo in enumerate(self.env.unwrapped.button_map)
            }
        else:
            self._vizdoom_vector_map = None

        return {k: v for k, v in mapping.items() if v is not None}

    def _get_atari_action(self):
        """Return the Discrete action index for Atari environments."""
        for key in self.current_keys:
            if key in self.key_to_action:
                return self.key_to_action[key]
        return self.noop_action

    def _get_vizdoom_action(self):
        """Return the MultiBinary action vector for VizDoom environments."""
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

        for key, name in VIZDOOM_KEY_BINDINGS.items():
            if key in pressed:
                if key == pygame.K_LEFT:
                    press("MOVE_LEFT" if alt else name)
                elif key == pygame.K_RIGHT:
                    press("MOVE_RIGHT" if alt else name)
                else:
                    press(name)

        space = self.env.action_space
        # When the action space contains both binary and continuous components
        # (e.g. Dict["binary", "continuous"]), build the appropriate dict
        if isinstance(space, gym.spaces.Dict):
            binary_space = space.get("binary")
            continuous_space = space.get("continuous")
            if isinstance(binary_space, gym.spaces.Discrete):
                if self._vizdoom_vector_map is None:
                    self._vizdoom_vector_map = {
                        tuple(c): i for i, c in enumerate(self.env.unwrapped.button_map)
                    }
                binary_action = self._vizdoom_vector_map.get(tuple(action), 0)
            else:
                binary_action = action

            if continuous_space is not None:
                cont_shape = continuous_space.shape
                continuous_action = np.zeros(cont_shape, dtype=np.float32)
                return {"binary": binary_action, "continuous": continuous_action}
            return {"binary": binary_action}

        if isinstance(space, gym.spaces.Discrete):
            if self._vizdoom_vector_map is None:
                self._vizdoom_vector_map = {
                    tuple(c): i for i, c in enumerate(self.env.unwrapped.button_map)
                }
            return self._vizdoom_vector_map.get(tuple(action), 0)

        return action

    def _get_stable_retro_action(self):
        """Return the MultiBinary action vector for stable-retro environments."""
        action = np.zeros(self.env.action_space.n, dtype=np.int32)
        platform = getattr(self.env.unwrapped, "system", None)
        mapping = STABLE_RETRO_KEY_BINDINGS.get(platform, {})
        for key in self.current_keys:
            idx = mapping.get(key)
            if idx is not None and idx < action.shape[0]:
                action[idx] = 1
        return action

    def _get_user_action(self):
        """Map pressed keys to actions for the current environment."""
        with self.key_lock:
            if hasattr(self.env, '_vizdoom') and self.env._vizdoom:
                return self._get_vizdoom_action()
            if hasattr(self.env, '_stable_retro') and self.env._stable_retro:
                return self._get_stable_retro_action()
            return self._get_atari_action()

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

    def _wait_for_start(self, start_key: int = START_KEY) -> bool:
        """Display overlay prompting the user to start.

        Returns True if the start key was pressed, False if the user closed the
        window or pressed ESC.
        """
        if self.screen is None:
            return True

        font = pygame.font.Font(None, 48)
        text = font.render("Press SPACE to start", True, (255, 255, 255))
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))

        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == start_key:
                        return True
            # Redraw overlay each frame
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(30)

    async def record(self, fps=None):
        """Record a gameplay session at the desired FPS."""
        if fps is None:
            fps = get_default_fps(self.env)
        self.recording = True
        try: return await self._record(fps=fps)
        finally: self.recording = False

    async def _record(self, fps=None):
        self.recording = True
        try:
            await self.play(fps=fps)
            data = {
                "episode_id": self.episode_ids,
                "timestamp": self.timestamps,
                "image": self.frames,
                "step": self.steps,
                "action": self.actions,
            }
            dataset = Dataset.from_dict(data)
            dataset = dataset.cast_column("image", HFImage())
            return dataset
        finally: 
            self.recording = False

    async def play(self, fps=None):
        if fps is None:
            fps = get_default_fps(self.env)
        try: await self._play(fps)
        finally: self.close()

    async def _play(self, fps=None):
        """
        Main loop for interactive gameplay and recording.
        """
        if fps is None:
            fps = get_default_fps(self.env)
        clock = pygame.time.Clock()
        
        self.episode_ids.clear()
        self.frames.clear()
        self.actions.clear()
        self.steps.clear()
        self.timestamps.clear()

        episode_id = int(time.time())
        obs, _ = self.env.reset()
        self._ensure_screen(obs)  # Ensure pygame window is created after first obs
        self._render_frame(obs)
        if not self._wait_for_start():
            return
        with self.key_lock:
            self.current_keys.clear()
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

    async def replay(self, actions, fps=None):
        if fps is None:
            fps = get_default_fps(self.env)
        clock = pygame.time.Clock()
        obs, _ = self.env.reset()
        self._ensure_screen(obs)
        self._render_frame(obs)
        for action in tqdm(actions):
            # Convert stored actions back to the environment's expected format
            if isinstance(action, list):
                if isinstance(self.env.action_space, gym.spaces.Discrete) and len(action) == 1:
                    action = action[0]
                else:
                    action = np.array(action, dtype=np.int32)
            elif isinstance(action, dict):
                new_action = {}
                space = self.env.action_space
                if isinstance(space, gym.spaces.Dict):
                    for k, v in action.items():
                        sub = space[k]
                        if isinstance(v, list):
                            new_action[k] = np.array(v, dtype=sub.dtype)
                        else:
                            new_action[k] = v
                    action = new_action
                else:
                    for k, v in action.items():
                        new_action[k] = np.array(v) if isinstance(v, list) else v
                    action = new_action
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


def generate_dataset_card(dataset, env_id, repo_id):
    """Generate or update the dataset card for a given dataset repo."""

    # Dataset statistics
    frames = len(dataset)
    episodes = len(set(dataset["episode_id"]))

    structure_lines = []
    for name, feature in dataset.features.items():
        structure_lines.append(f"- **{name}**: {feature}")
    dataset_structure = "This dataset contains the following columns:\n" + "\n".join(structure_lines)

    dataset_summary = (
        f"This dataset contains {frames} frames recorded from the Gymnasium environment "
        f"`{env_id}` across {episodes} episodes."
    )

    user_info = whoami()
    curator = user_info.get("name") or user_info.get("user") or "unknown"

    card_data = DatasetCardData(
        language="en",
        license="mit",
        task_categories=["reinforcement-learning"],
        pretty_name=f"{env_id} Gameplay Dataset",
    )

    # Build card content manually to avoid placeholder fields
    header = card_data.to_yaml()
    content_lines = [
        "---",
        header,
        "---",
        "",
        f"# {card_data.pretty_name}",
        "",
        dataset_summary,
        "",
        f"Environment ID: `{env_id}`",
        "",
        "## Dataset Structure",
        dataset_structure,
        "",
        f"Curated by: {curator}",
    ]
    card = DatasetCard("\n".join(content_lines))

    card.push_to_hub(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )

def create_env(env_id):
    # In case ROM is suffixed with platform name then use stable-retro
    retro_platforms = {
        "Nes",
        "Atari2600",
        "Snes",
        "GbAdvance",
        "GameBoy",
        "GbColor",
        "PCEngine",
        "Saturn",
        "32x",
        "Genesis",
        "Sms",
        "GameGear",
        "SCD",
    }
    match = re.search(r"-(" + "|".join(retro_platforms) + r")$", env_id)
    if match:
        import retro
        #platform = match.group(1)
        #game_name = env_id.replace(f"-{platform}", "")
        env = retro.make(env_id, render_mode="rgb_array")
        env._stable_retro = True
    elif "Vizdoom" in env_id:
        from vizdoom import gymnasium_wrapper
        # use MultiBinary actions to avoid losing button presses
        env = gym.make(env_id, render_mode="rgb_array", max_buttons_pressed=0)
        env._vizdoom = True
    # Otherwise use ale_py
    else:
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make(env_id, render_mode="rgb_array")
    
    env._env_id = env_id.replace("-", "_")
    return env

def get_default_fps(env):
    """Determine a sensible default FPS for an environment."""

    # First try to read FPS information from the environment metadata. This is
    # provided by most Gymnasium environments.  gym-retro/stable-retro exposes
    # the console framerate under ``video.frames_per_second`` while Atari/VizDoom
    # and others use ``render_fps``.
    for key in ("render_fps", "video.frames_per_second"):
        fps = env.metadata.get(key)
        if fps:
            try:
                return int(round(float(fps)))
            except (TypeError, ValueError):
                pass

    # Fall back to heuristics based on the environment id when metadata is not
    # available.  These values are approximate but keep gameplay reasonable.
    env_id = getattr(env, "_env_id", "")

    # Some retro consoles commonly run at 60 FPS
    retro_platforms_60fps = {
        "Nes",
        "GameBoy",
        "Snes",
        "GbAdvance",
        "GbColor",
        "Genesis",
        "PCEngine",
        "Saturn",
        "32x",
        "Sms",
        "GameGear",
        "SCD",
    }
    if any(env_id.endswith(f"-{plat}") for plat in retro_platforms_60fps) or "Atari2600" in env_id:
        return 60

    if "Vizdoom" in env_id or "vizdoom" in env_id:
        return 35

    # Atari via ALE is often played at a reduced 15 FPS for humans
    if "NoFrameskip" in env_id or "ALE" in env_id:
        return 15

    return 15

def list_environments():
    """Print available Atari, stable-retro and VizDoom environments."""
    print("=== Atari Environments ===")
    try:
        import ale_py
        gym.register_envs(ale_py)
        atari_ids = sorted(
            env_id
            for env_id in gym.envs.registry.keys()
            if str(gym.spec(env_id).entry_point) == "ale_py.env:AtariEnv"
        )
        for env_id in atari_ids:
            print(env_id)
    except Exception as e:
        print(f"Could not list Atari environments: {e}")

    try:
        import retro
        all_games = sorted(retro.data.list_games(retro.data.Integrations.ALL))
        if all_games:
            print("\n=== Stable-Retro Games ===")
            for game in all_games:
                try:
                    retro.data.get_romfile_path(game, retro.data.Integrations.ALL)
                    status = "(imported)"
                except FileNotFoundError:
                    status = "(missing ROM)"
                print(f"{game} {status}")
        else:
            print("\nStable-Retro package installed but no games found.")
    except Exception as e:
        print(f"\nStable-Retro not installed: {e}")

    try:
        import vizdoom
        import vizdoom.gymnasium_wrapper  # register gym environments
        vizdoom_ids = [
            env_id for env_id in gym.envs.registry.keys() if env_id.startswith("Vizdoom")
        ]
        if vizdoom_ids:
            print("\n=== VizDoom Environments ===")
            for env_id in vizdoom_ids:
                print(env_id)
        if getattr(vizdoom, "wads", None):
            print("\nAvailable WADs:")
            for wad in vizdoom.wads:
                print(wad)
    except Exception:
        print("\nVizDoom not installed.")

async def main():
    parser = argparse.ArgumentParser(description="Atari Gymnasium Recorder/Playback")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_record = subparsers.add_parser("record", help="Record gameplay")
    parser_record.add_argument("env_id", type=str, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")
    parser_record.add_argument("--fps", type=int, default=None, help="Frames per second for playback/recording")

    parser_playback = subparsers.add_parser("playback", help="Replay a dataset")
    parser_playback.add_argument("env_id", type=str, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")
    parser_playback.add_argument("--fps", type=int, default=None, help="Frames per second for playback/recording")

    subparsers.add_parser("list_environments", help="List available environments")

    args = parser.parse_args()

    if args.command == "list_environments":
        list_environments()
        return

    env_id = args.env_id
    hf_repo_id = env_id_to_hf_repo_id(env_id)
    loaded_dataset = None
    api = HfApi()
    try:
        api.dataset_info(hf_repo_id)
        loaded_dataset = load_dataset(
            hf_repo_id,
            split="train",
            download_mode="force_redownload",
        )
        if isinstance(loaded_dataset.features["action"], Value):
            loaded_dataset = loaded_dataset.map(lambda row: {"action": [row["action"]]})
            loaded_dataset = loaded_dataset.cast_column("action", Sequence(Value("int64")))
    except Exception:
        loaded_dataset = None

    env = create_env(env_id)
    fps = args.fps if args.fps is not None else get_default_fps(env)

    if args.command == "record":
        recorder = DatasetRecorderWrapper(env)
        recorded_dataset = await recorder.record(fps=fps)
        final_dataset = (
            concatenate_datasets([loaded_dataset, recorded_dataset]) if loaded_dataset else recorded_dataset
        )
        upload = input("Upload dataset to Hugging Face Hub? [Y/n] ").strip().lower()
        if upload in ("", "y", "yes"):
            final_dataset.push_to_hub(hf_repo_id)
            generate_dataset_card(final_dataset, env_id, hf_repo_id)
        else:
            print("Skipping upload. Dataset was not pushed to the hub.")
    elif args.command == "playback":
        assert loaded_dataset is not None, f"Dataset not found: {hf_repo_id}"
        recorder = DatasetRecorderWrapper(env)
        actions = loaded_dataset["action"]
        await recorder.replay(actions, fps=fps)

if __name__ == "__main__":
    asyncio.run(main())
