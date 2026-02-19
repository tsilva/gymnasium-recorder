import os
import sys
import time
import json
import shutil
import threading
import asyncio
import tempfile
import argparse
import tomllib
import uuid

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

console = Console()

STYLE_KEY = "bold cyan"
STYLE_ACTION = "bold white"
STYLE_ENV = "bold green"
STYLE_PATH = "dim"
STYLE_CMD = "bold yellow"
STYLE_SUCCESS = "bold green"
STYLE_FAIL = "bold red"
STYLE_INFO = "cyan"

from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file

import gymnasium as gym

_initialized = False


def _json_default(obj):
    """JSON serializer for numpy types found in info dicts."""
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return str(obj)


def _build_key_name_map(pygame):
    """Build a mapping from human-readable key names to pygame key constants."""
    key_map = {
        "up": pygame.K_UP,
        "down": pygame.K_DOWN,
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "space": pygame.K_SPACE,
        "tab": pygame.K_TAB,
        "return": pygame.K_RETURN,
        "lshift": pygame.K_LSHIFT,
        "rshift": pygame.K_RSHIFT,
        "lctrl": pygame.K_LCTRL,
        "rctrl": pygame.K_RCTRL,
    }
    for c in "abcdefghijklmnopqrstuvwxyz":
        key_map[c] = getattr(pygame, f"K_{c}")
    for d in "0123456789":
        key_map[d] = getattr(pygame, f"K_{d}")
    return key_map


def _resolve_key(name, key_map):
    """Resolve a human-readable key name to a pygame constant."""
    name_lower = name.lower()
    if name_lower not in key_map:
        raise ValueError(
            f"Unknown key name '{name}' in keymappings.toml. "
            f"Valid keys: {', '.join(sorted(key_map.keys()))}"
        )
    return key_map[name_lower]


def _load_keymappings(pygame):
    """Load keymappings from keymappings.toml, falling back to defaults."""
    key_map = _build_key_name_map(pygame)

    # Build defaults (same values as previously hardcoded)
    default_start_key = pygame.K_SPACE

    default_atari = {
        pygame.K_SPACE: "FIRE",
        pygame.K_UP: "UP",
        pygame.K_RIGHT: "RIGHT",
        pygame.K_LEFT: "LEFT",
        pygame.K_DOWN: "DOWN",
    }

    default_vizdoom = {
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
        default_vizdoom[getattr(pygame, f"K_{i}")] = f"SELECT_WEAPON{i}"

    default_retro = {
        "Nes": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "Atari2600": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
        },
        "Snes": {
            pygame.K_z: 0,
            pygame.K_a: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
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
            pygame.K_n: 2,
            pygame.K_m: 3,
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
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "GbColor": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "PCEngine": {
            pygame.K_x: 0,
            pygame.K_c: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
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
            pygame.K_n: 2,
            pygame.K_m: 3,
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
            pygame.K_n: 2,
            pygame.K_m: 3,
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
            pygame.K_n: 2,
            pygame.K_m: 3,
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
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "GameGear": {
            pygame.K_z: 0,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "SCD": {
            pygame.K_x: 0,
            pygame.K_z: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
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

    # Try loading config file
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "keymappings.toml"
    )
    if not os.path.exists(config_path):
        return default_start_key, default_atari, default_vizdoom, default_retro

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # General section
    start_key = default_start_key
    if "general" in config and "start_key" in config["general"]:
        start_key = _resolve_key(config["general"]["start_key"], key_map)

    # Atari section
    atari = default_atari
    if "atari" in config:
        atari = {}
        for key_name, action in config["atari"].items():
            atari[_resolve_key(key_name, key_map)] = action

    # VizDoom section
    vizdoom = default_vizdoom
    if "vizdoom" in config:
        vizdoom = {}
        for key_name, action in config["vizdoom"].items():
            vizdoom[_resolve_key(key_name, key_map)] = action

    # Stable-Retro section
    retro = default_retro
    if "stable_retro" in config:
        retro = {}
        for console, bindings in config["stable_retro"].items():
            retro[console] = {}
            for key_name, action in bindings.items():
                retro[console][_resolve_key(key_name, key_map)] = action

    return start_key, atari, vizdoom, retro


DEFAULT_CONFIG = {
    "display": {"scale_factor": 2},
    "fps_defaults": {"atari": 90, "vizdoom": 45, "retro": 90},
    "dataset": {
        "repo_prefix": "gymrec__",
        "license": "mit",
        "task_categories": ["reinforcement-learning"],
        "commit_message": "Update dataset card",
    },
    "storage": {
        "local_dir": os.path.join(os.path.expanduser("~"), ".gymrec", "datasets"),
    },
    "overlay": {
        "font_size": 48,
        "text_color": [255, 255, 255],
        "background_color": [0, 0, 0, 180],
        "fps": 30,
        "text": "Press SPACE to start",
    },
}

CONFIG = None


def _load_config():
    """Load configuration from config.toml, falling back to defaults."""
    import copy

    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.toml"
    )
    if not os.path.exists(config_path):
        return config
    with open(config_path, "rb") as f:
        user_config = tomllib.load(f)
    for section in config:
        if section in user_config:
            config[section].update(user_config[section])
    return config


def _lazy_init():
    """Import heavy dependencies and initialize key bindings on first use."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    global CONFIG
    global np, pygame, PILImage
    global whoami, DatasetCard, DatasetCardData, HfApi, login, get_token
    global Dataset, HFImage, load_dataset, load_from_disk, load_dataset_builder
    global \
        START_KEY, \
        ATARI_KEY_BINDINGS, \
        VIZDOOM_KEY_BINDINGS, \
        STABLE_RETRO_KEY_BINDINGS

    import numpy as np
    import pygame
    from PIL import Image as PILImage
    from huggingface_hub import (
        whoami,
        DatasetCard,
        DatasetCardData,
        HfApi,
        login,
        get_token,
    )
    from datasets import (
        Dataset,
        Image as HFImage,
        load_dataset,
        load_from_disk,
        load_dataset_builder,
    )

    START_KEY, ATARI_KEY_BINDINGS, VIZDOOM_KEY_BINDINGS, STABLE_RETRO_KEY_BINDINGS = (
        _load_keymappings(pygame)
    )
    CONFIG = _load_config()


def ensure_hf_login(force=False) -> bool:
    """Ensure user is logged in to Hugging Face Hub. Prompts interactively if needed."""
    _lazy_init()
    token = get_token()

    if token and not force:
        return True

    if token and force:
        try:
            info = whoami(token=token)
            username = info.get("name", "unknown")
            console.print(
                f"[{STYLE_SUCCESS}]Already logged in as [{STYLE_ENV}]{username}[/][/]"
            )
            if not Confirm.ask("Re-login with a different token?", default=False):
                return True
        except Exception:
            console.print(f"[{STYLE_FAIL}]Existing token is invalid.[/]")

    console.print(
        Panel(
            f"[{STYLE_ACTION}]Create a token at:[/] [{STYLE_CMD}]https://huggingface.co/settings/tokens[/]\n"
            f"Required permission: [{STYLE_INFO}]write[/]",
            title="Hugging Face Login",
            border_style="cyan",
        )
    )

    for attempt in range(1, 4):
        token_input = Prompt.ask("Paste your token", password=True)
        if not token_input.strip():
            console.print(f"[{STYLE_FAIL}]Empty token, try again.[/]")
            continue
        try:
            login(token=token_input.strip())
            info = whoami()
            username = info.get("name", "unknown")
            console.print(
                f"[{STYLE_SUCCESS}]Logged in as [{STYLE_ENV}]{username}[/][/]"
            )
            return True
        except Exception as e:
            remaining = 3 - attempt
            if remaining > 0:
                console.print(
                    f"[{STYLE_FAIL}]Login failed: {e}[/] ({remaining} attempt{'s' if remaining > 1 else ''} left)"
                )
            else:
                console.print(f"[{STYLE_FAIL}]Login failed: {e}[/]")

    console.print(
        f"[{STYLE_FAIL}]Could not authenticate. Try:[/] [{STYLE_CMD}]uv run python main.py login[/]"
    )
    return False


class DatasetRecorderWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for recording and replaying Atari gameplay as Hugging Face datasets.
    """

    def __init__(self, env):
        _lazy_init()
        super().__init__(env)

        self.recording = False
        self.frame_shape = None  # Delay initialization
        self.screen = None  # Delay initialization

        pygame.init()
        # pygame.display.set_caption will be set after env_id is available

        self.current_keys = set()
        self.key_lock = threading.Lock()
        self.key_to_action = None  # Resolved lazily in _resolve_atari_key_mapping
        self._atari_key_bindings_raw = ATARI_KEY_BINDINGS
        self._vizdoom_buttons = None
        self._vizdoom_vector_map = None
        self.noop_action = 0

        self.episode_ids = []
        self.seeds = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.terminations = []
        self.truncations = []
        self.infos = []
        self._current_episode_uuid = None

        self.temp_dir = tempfile.mkdtemp()

        self._fps = None
        self._fps_changed_at = 0
        self._episode_count = 0
        self._cumulative_reward = 0.0
        self._overlay_visible = True
        self._recorded_dataset = None
        self._env_metadata = None

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
            scale = CONFIG["display"]["scale_factor"]
            self.screen = pygame.display.set_mode(
                (self.frame_shape[1] * scale, self.frame_shape[0] * scale)
            )
            pygame.display.set_caption(
                getattr(self.env, "_env_id", "Gymnasium Recorder")
            )

    def _save_frame_image(self, frame):
        """Save a frame as lossless WebP and return the file path."""
        if isinstance(frame, dict):
            for k in ["obs", "image", "screen"]:
                if k in frame:
                    frame = frame[k]
                    break
        frame_uint8 = frame.astype(np.uint8)
        path = os.path.join(self.temp_dir, f"frame_{len(self.frames):05d}.webp")
        img = PILImage.fromarray(frame_uint8)
        img.save(path, format="WEBP", lossless=True, method=6)
        return path

    @staticmethod
    def _normalize_action(action):
        """Normalize action format for dataset storage."""
        if isinstance(action, np.ndarray):
            return action.tolist()
        elif isinstance(action, dict):
            return {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in action.items()
            }
        else:
            return [int(action)]

    def _record_frame(self, episode_uuid, seed, step, frame, action):
        """Save a frame and action to temporary storage."""
        if not self.recording:
            return

        path = self._save_frame_image(frame)
        self.episode_ids.append(episode_uuid.bytes)
        self.seeds.append(seed if step == 0 else None)
        self.frames.append(path)
        self.actions.append(self._normalize_action(action))

    def _record_terminal_observation(self, episode_uuid, frame):
        """Record a terminal observation row (Minari N+1 pattern).

        The N+1 observation captures the final state after the last step.
        It has an empty action and null values for reward/termination/truncation/info
        since no step was taken.
        """
        if not self.recording:
            return

        path = self._save_frame_image(frame)
        self.episode_ids.append(episode_uuid.bytes)
        self.seeds.append(None)
        self.frames.append(path)
        self.actions.append([])
        self.rewards.append(None)
        self.terminations.append(None)
        self.truncations.append(None)
        self.infos.append(None)

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
                if event.key == pygame.K_TAB:
                    self._overlay_visible = not self._overlay_visible
                    continue
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    if self._fps is not None:
                        self._fps = max(1, self._fps + 5)
                        self._fps_changed_at = time.monotonic()
                    continue
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    if self._fps is not None:
                        self._fps = max(1, self._fps - 5)
                        self._fps_changed_at = time.monotonic()
                    continue
                with self.key_lock:
                    self.current_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                with self.key_lock:
                    self.current_keys.discard(event.key)
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

    def _resolve_atari_key_mapping(self):
        """Resolve meaning-based Atari key bindings to action indices using the env's action meanings."""
        # Standard fallback: meaning -> index for the default Atari action order
        standard_meaning_to_idx = {
            "NOOP": 0,
            "FIRE": 1,
            "UP": 2,
            "RIGHT": 3,
            "LEFT": 4,
            "DOWN": 5,
            "UPRIGHT": 6,
            "UPLEFT": 7,
            "DOWNRIGHT": 8,
            "DOWNLEFT": 9,
            "UPFIRE": 10,
            "RIGHTFIRE": 11,
            "LEFTFIRE": 12,
            "DOWNFIRE": 13,
            "UPRIGHTFIRE": 14,
            "UPLEFTFIRE": 15,
            "DOWNRIGHTFIRE": 16,
            "DOWNLEFTFIRE": 17,
        }

        # Try to get actual meanings from the environment
        meaning_to_idx = None
        try:
            meanings = self.env.unwrapped.get_action_meanings()
            if meanings:
                meaning_to_idx = {}
                for idx, m in enumerate(meanings):
                    meaning_to_idx[m.upper()] = idx
        except (AttributeError, TypeError):
            pass

        if meaning_to_idx is None:
            meaning_to_idx = standard_meaning_to_idx

        resolved = {}
        for key, value in self._atari_key_bindings_raw.items():
            if isinstance(value, str):
                idx = meaning_to_idx.get(value.upper())
                if idx is not None:
                    resolved[key] = idx

        self.key_to_action = resolved
        self._atari_meaning_to_idx = meaning_to_idx
        # Reverse map: pygame key -> meaning string (for composite action lookup)
        self._atari_key_to_meaning = {}
        for key, value in self._atari_key_bindings_raw.items():
            if isinstance(value, str):
                self._atari_key_to_meaning[key] = value.upper()

    def _get_atari_action(self):
        """Return the Discrete action index for Atari environments."""
        if self.key_to_action is None:
            self._resolve_atari_key_mapping()

        # Collect all pressed meaning strings
        pressed_meanings = set()
        for key in self.current_keys:
            if key in self._atari_key_to_meaning:
                pressed_meanings.add(self._atari_key_to_meaning[key])

        if not pressed_meanings:
            return self.noop_action

        # Build composite name following ALE convention: [UP|DOWN][RIGHT|LEFT][FIRE]
        composite = ""
        if "UP" in pressed_meanings:
            composite += "UP"
        elif "DOWN" in pressed_meanings:
            composite += "DOWN"
        if "RIGHT" in pressed_meanings:
            composite += "RIGHT"
        elif "LEFT" in pressed_meanings:
            composite += "LEFT"
        if "FIRE" in pressed_meanings:
            composite += "FIRE"

        if not composite:
            return self.noop_action

        # Try composite first, fall back to single key
        if composite in self._atari_meaning_to_idx:
            return self._atari_meaning_to_idx[composite]

        # Fallback: return first matching single key
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
            if hasattr(self.env, "_vizdoom") and self.env._vizdoom:
                return self._get_vizdoom_action()
            if hasattr(self.env, "_stable_retro") and self.env._stable_retro:
                return self._get_stable_retro_action()
            return self._get_atari_action()

    def _render_frame(self, frame):
        """
        Render a frame using pygame, scaled by the configured scale factor.
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

        scale = CONFIG["display"]["scale_factor"]
        w, h = surface.get_size()
        scaled_surface = pygame.transform.scale(surface, (w * scale, h * scale))

        # Update display with scaled frame
        self.screen.blit(scaled_surface, (0, 0))
        if self._overlay_visible:
            self._render_fps_overlay()
            self._render_episode_overlay()
        pygame.display.flip()

    def _render_fps_overlay(self):
        """Render a temporary FPS indicator in the top-right corner."""
        if self._fps is None or self.screen is None:
            return
        elapsed = time.monotonic() - self._fps_changed_at
        if elapsed >= 1.5:
            return

        # Compute alpha: full opacity for first 1.0s, fade over last 0.5s
        if elapsed < 1.0:
            alpha = 255
        else:
            alpha = int(255 * (1.5 - elapsed) / 0.5)

        font = pygame.font.Font(None, 24)
        text = font.render(f"{self._fps} FPS", True, (255, 255, 255))
        text_rect = text.get_rect()

        padding = 6
        bg_w = text_rect.width + padding * 2
        bg_h = text_rect.height + padding * 2
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, min(alpha, 180)))

        text_alpha = pygame.Surface(text_rect.size, pygame.SRCALPHA)
        text_alpha.blit(text, (0, 0))
        text_alpha.set_alpha(alpha)

        screen_w = self.screen.get_width()
        bg_x = screen_w - bg_w - 8
        bg_y = 8
        self.screen.blit(bg, (bg_x, bg_y))
        self.screen.blit(text_alpha, (bg_x + padding, bg_y + padding))

    def _render_episode_overlay(self):
        """Render a persistent HUD badge in the top-left corner during recording."""
        if not self.recording or self._episode_count < 1 or self.screen is None:
            return

        parts = [f"EP {self._episode_count}", f"R {self._cumulative_reward:.0f}"]
        if self._fps is not None:
            parts.append(f"{self._fps} FPS")
        font = pygame.font.Font(None, 24)
        text = font.render("  ".join(parts), True, (255, 255, 255))
        text_rect = text.get_rect()

        padding = 6
        bg_w = text_rect.width + padding * 2
        bg_h = text_rect.height + padding * 2
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))

        bg_x = 8
        bg_y = 8
        self.screen.blit(bg, (bg_x, bg_y))
        self.screen.blit(text, (bg_x + padding, bg_y + padding))

    def _print_keymappings(self):
        """Print the current key mappings to the console."""
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Key", justify="right", style=STYLE_KEY)
        table.add_column("Action", style=STYLE_ACTION)
        table.add_column("Index", style=STYLE_PATH)

        if hasattr(self.env, "_vizdoom") and self.env._vizdoom:
            env_type = "VizDoom"
            if self._vizdoom_buttons is None:
                self._vizdoom_buttons = self._init_vizdoom_key_mapping()
            for key, action_name in VIZDOOM_KEY_BINDINGS.items():
                btn_idx = self._vizdoom_buttons.get(action_name)
                idx_str = f"btn {btn_idx}" if btn_idx is not None else ""
                table.add_row(pygame.key.name(key), action_name, idx_str)
            ml_idx = self._vizdoom_buttons.get("MOVE_LEFT")
            mr_idx = self._vizdoom_buttons.get("MOVE_RIGHT")
            table.add_row(
                "alt+left", "MOVE_LEFT", f"btn {ml_idx}" if ml_idx is not None else ""
            )
            table.add_row(
                "alt+right", "MOVE_RIGHT", f"btn {mr_idx}" if mr_idx is not None else ""
            )
        elif hasattr(self.env, "_stable_retro") and self.env._stable_retro:
            platform = getattr(self.env.unwrapped, "system", None)
            env_type = f"Stable-Retro ({platform})"
            buttons = getattr(self.env.unwrapped, "buttons", None)
            mapping = STABLE_RETRO_KEY_BINDINGS.get(platform, {})
            # Group keys: D-pad first, then action buttons, then special (SELECT/START)
            dpad_keys = {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT}
            special_labels = {"SELECT", "START"}
            group_dpad = []
            group_action = []
            group_special = []
            for key, idx in mapping.items():
                label = (
                    buttons[idx] if buttons and idx < len(buttons) else f"button {idx}"
                )
                row = (pygame.key.name(key), label, f"idx {idx}")
                if key in dpad_keys:
                    group_dpad.append(row)
                elif label.upper() in special_labels:
                    group_special.append(row)
                else:
                    group_action.append(row)
            for row in group_dpad:
                table.add_row(*row)
            if group_dpad and group_action:
                table.add_section()
            for row in group_action:
                table.add_row(*row)
            if group_action and group_special:
                table.add_section()
            for row in group_special:
                table.add_row(*row)
        else:
            env_type = "Atari"
            if self.key_to_action is None:
                self._resolve_atari_key_mapping()
            try:
                meanings = self.env.unwrapped.get_action_meanings()
            except (AttributeError, TypeError):
                meanings = None
            for key, action_idx in self.key_to_action.items():
                label = (
                    meanings[action_idx]
                    if meanings and action_idx < len(meanings)
                    else f"action {action_idx}"
                )
                table.add_row(pygame.key.name(key), label, f"action {action_idx}")

        table.add_section()
        table.add_row("[dim]escape[/]", "[dim]Exit[/]", "")
        table.add_row("[dim]+/-[/]", "[dim]Adjust FPS (±5)[/]", "")

        console.print(
            Panel(
                table,
                title=f"[{STYLE_ENV}]{env_type}[/] Key Mappings",
                border_style=STYLE_INFO,
                expand=False,
            )
        )

    def _wait_for_start(self, start_key: int = None) -> bool:
        """Display overlay prompting the user to start.

        Returns True if the start key was pressed, False if the user closed the
        window or pressed ESC.
        """
        if start_key is None:
            start_key = START_KEY
        if self.screen is None:
            return True

        overlay_cfg = CONFIG["overlay"]
        font = pygame.font.Font(None, overlay_cfg["font_size"])
        text = font.render(overlay_cfg["text"], True, tuple(overlay_cfg["text_color"]))
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill(tuple(overlay_cfg["background_color"]))
        text_rect = text.get_rect(
            center=(self.screen.get_width() // 2, self.screen.get_height() // 2)
        )

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
            clock.tick(overlay_cfg["fps"])

    async def record(self, fps=None):
        """Record a gameplay session at the desired FPS."""
        if fps is None:
            fps = get_default_fps(self.env)
        self.recording = True
        try:
            return await self._record(fps=fps)
        finally:
            self.recording = False

    async def _record(self, fps=None):
        try:
            await self._play(fps)  # bypass play() to avoid premature close()
            return self._recorded_dataset
        finally:
            # Don't delete temp_dir here - dataset.save_to_disk() needs the image files
            # temp_dir cleanup happens in main() after save_dataset_locally()
            pygame.quit()
            self.env.close()

    async def _play(self, fps=None):
        """
        Main loop for interactive gameplay and recording.
        """
        if fps is None:
            fps = get_default_fps(self.env)
        self._fps = fps

        self.episode_ids.clear()
        self.seeds.clear()
        self.frames.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.infos.clear()

        # Capture environment metadata for dataset card
        self._env_metadata = _capture_env_metadata(self.env)

        self._current_episode_uuid = uuid.uuid4()
        seed = int(time.time())
        self._episode_count = 1
        self._cumulative_reward = 0.0
        obs, _ = self.env.reset(seed=seed)
        self._ensure_screen(obs)  # Ensure pygame window is created after first obs
        self._render_frame(obs)
        self._print_keymappings()
        if not self._wait_for_start():
            return
        with self.key_lock:
            self.current_keys.clear()
        step = 0
        while True:
            frame_start = time.monotonic()
            if not self._input_loop():
                break
            action = self._get_user_action()
            self._record_frame(self._current_episode_uuid, seed, step, obs, action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._cumulative_reward += float(reward)
            if self.recording:
                self.rewards.append(float(reward))
                self.terminations.append(bool(terminated))
                self.truncations.append(bool(truncated))
                self.infos.append(json.dumps(info, default=_json_default))
            self._render_frame(obs)
            elapsed = time.monotonic() - frame_start
            remaining = (1.0 / self._fps) - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            else:
                await asyncio.sleep(0)
            step += 1

            if terminated or truncated:
                self._record_terminal_observation(self._current_episode_uuid, obs)
                self._current_episode_uuid = uuid.uuid4()
                seed = int(time.time())
                obs, _ = self.env.reset(seed=seed)
                self._episode_count += 1
                self._cumulative_reward = 0.0
                step = 0
                self._render_frame(obs)

        # Record terminal observation on user exit (ESC)
        if self.recording and self.frames and self.actions[-1] != []:
            # Mark last real step as truncated: user exited mid-episode.
            # Minari requires at least one True in terminations or truncations per episode.
            self.truncations[-1] = True
            self._record_terminal_observation(self._current_episode_uuid, obs)

        if self.recording and self.frames:
            import pyarrow as pa

            data = {
                "episode_id": self.episode_ids,
                "seed": self.seeds,
                "observations": self.frames,
                "actions": self.actions,
                "rewards": self.rewards,
                "terminations": self.terminations,
                "truncations": self.truncations,
                "infos": self.infos,
            }
            self._recorded_dataset = Dataset.from_dict(data)
            self._recorded_dataset = self._recorded_dataset.cast_column(
                "observations", HFImage()
            )

    def _extract_obs_image(self, obs):
        """Extract image array from observation, handling dict obs (e.g. VizDoom)."""
        if isinstance(obs, dict):
            for k in ["obs", "image", "screen"]:
                if k in obs:
                    return obs[k]
        return obs

    def _convert_action(self, action):
        """Convert stored action back to the environment's expected format."""
        if isinstance(action, list):
            if (
                isinstance(self.env.action_space, gym.spaces.Discrete)
                and len(action) == 1
            ):
                return action[0]
            else:
                return np.array(action, dtype=np.int32)
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
            else:
                for k, v in action.items():
                    new_action[k] = np.array(v) if isinstance(v, list) else v
            return new_action
        return action

    async def replay(self, actions, fps=None, total=None, verify=False):
        if fps is None:
            fps = get_default_fps(self.env)
        self._fps = fps
        obs, _ = self.env.reset()
        self._ensure_screen(obs)
        self._render_frame(obs)
        self._print_keymappings()

        mse_threshold = 5.0
        verify_metrics = [] if verify else None
        reward_mismatches = 0
        terminal_mismatches = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task("Replaying", total=total)
            for item in actions:
                frame_start = time.monotonic()
                if not self._input_loop():
                    break

                if verify:
                    (
                        action,
                        recorded_image,
                        recorded_reward,
                        recorded_terminated,
                        recorded_truncated,
                    ) = item
                else:
                    action = item

                action = self._convert_action(action)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                self._render_frame(obs)

                if verify:
                    obs_image = self._extract_obs_image(obs)
                    if obs_image.dtype != np.uint8:
                        obs_image = obs_image.astype(np.uint8)
                    recorded_array = np.array(recorded_image, dtype=np.uint8)

                    if obs_image.shape == recorded_array.shape:
                        mse = float(
                            np.mean(
                                (
                                    obs_image.astype(np.float32)
                                    - recorded_array.astype(np.float32)
                                )
                                ** 2
                            )
                        )
                    else:
                        console.print(
                            f"  [yellow]Warning:[/] shape mismatch at frame {len(verify_metrics)}: "
                            f"obs={obs_image.shape} vs recorded={recorded_array.shape}, skipping comparison"
                        )
                        mse = None

                    if float(reward) != float(recorded_reward):
                        reward_mismatches += 1
                    if bool(terminated) != bool(recorded_terminated) or bool(
                        truncated
                    ) != bool(recorded_truncated):
                        terminal_mismatches += 1

                    verify_metrics.append(mse)

                elapsed = time.monotonic() - frame_start
                remaining = (1.0 / self._fps) - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                else:
                    await asyncio.sleep(0)
                progress.advance(ptask)

        if verify and verify_metrics:
            valid_mses = [m for m in verify_metrics if m is not None]
            n_total = len(verify_metrics)
            n_skipped = n_total - len(valid_mses)

            lines = [f"Total frames: [{STYLE_INFO}]{n_total}[/]"]
            if n_skipped > 0:
                lines.append(f"Skipped (shape mismatch): [yellow]{n_skipped}[/]")

            if valid_mses:
                mean_mse = sum(valid_mses) / len(valid_mses)
                max_mse = max(valid_mses)
                min_mse = min(valid_mses)
                exceeded = sum(1 for m in valid_mses if m > mse_threshold)
                lines.append(
                    f"Frame MSE:  mean=[{STYLE_INFO}]{mean_mse:.2f}[/], max=[{STYLE_INFO}]{max_mse:.2f}[/], min=[{STYLE_INFO}]{min_mse:.2f}[/]"
                )
                lines.append(
                    f"Reward mismatches: [{STYLE_INFO}]{reward_mismatches}/{n_total}[/] frames"
                )
                lines.append(
                    f"Terminal state mismatches: [{STYLE_INFO}]{terminal_mismatches}/{n_total}[/] frames"
                )

                passed = (
                    exceeded == 0
                    and reward_mismatches == 0
                    and terminal_mismatches == 0
                )
                if passed:
                    lines.append(
                        f"Result: [{STYLE_SUCCESS}]PASS[/] (all frames below threshold {mse_threshold})"
                    )
                    border_style = "green"
                else:
                    reasons = []
                    if exceeded > 0:
                        reasons.append(
                            f"{exceeded} frames exceeded MSE threshold {mse_threshold}"
                        )
                    if reward_mismatches > 0:
                        reasons.append(f"{reward_mismatches} reward mismatches")
                    if terminal_mismatches > 0:
                        reasons.append(
                            f"{terminal_mismatches} terminal state mismatches"
                        )
                    lines.append(
                        f"Result: [{STYLE_FAIL}]FAIL[/] ({', '.join(reasons)})"
                    )
                    border_style = "red"
            else:
                lines.append("[yellow]No valid frame comparisons (all skipped).[/]")
                border_style = "yellow"

            console.print()
            console.print(
                Panel(
                    "\n".join(lines),
                    title="Determinism Verification Report",
                    border_style=border_style,
                    expand=False,
                )
            )

    def close(self):
        """
        Clean up resources and save dataset if needed.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        pygame.quit()
        super().close()


def _encode_env_id_for_hf(env_id):
    """
    Encode env_id to a reversible string for HF dataset naming.

    Encoding scheme:
    - '/' -> '_slash_'
    - '-' -> '_dash_'
    - '_' -> '_underscore_'

    This allows perfect round-trip conversion between env_id and HF dataset name.
    """
    encoded = env_id.replace("_", "_underscore_")
    encoded = encoded.replace("-", "_dash_")
    encoded = encoded.replace("/", "_slash_")
    return encoded


def _decode_hf_repo_name(repo_name):
    """
    Decode HF dataset name back to original env_id.

    Reverse of _encode_env_id_for_hf().
    """
    # Must decode in reverse order to handle overlapping patterns correctly
    decoded = repo_name.replace("_slash_", "/")
    decoded = decoded.replace("_dash_", "-")
    decoded = decoded.replace("_underscore_", "_")
    return decoded


def env_id_to_hf_repo_id(env_id):
    user_info = whoami()
    username = (
        user_info.get("name") or user_info.get("user") or user_info.get("username")
    )
    encoded_env_id = _encode_env_id_for_hf(env_id)
    hf_repo_id = f"{username}/{CONFIG['dataset']['repo_prefix']}{encoded_env_id}"
    return hf_repo_id


def hf_repo_id_to_env_id(hf_repo_id):
    """Convert HF repo_id back to original env_id."""
    prefix = CONFIG["dataset"]["repo_prefix"]
    # Extract just the repo name part (after username/)
    if "/" in hf_repo_id:
        repo_name = hf_repo_id.split("/", 1)[1]
    else:
        repo_name = hf_repo_id

    if not repo_name.startswith(prefix):
        return None

    encoded_env_id = repo_name[len(prefix) :]
    return _decode_hf_repo_name(encoded_env_id)


def get_local_dataset_path(env_id):
    """Return the local storage path for a given environment's dataset."""
    encoded_env_id = _encode_env_id_for_hf(env_id)
    return os.path.join(CONFIG["storage"]["local_dir"], encoded_env_id)


def _get_available_envs_from_local():
    """Get list of env_ids that have local recordings."""
    local_dir = CONFIG["storage"]["local_dir"]
    if not os.path.exists(local_dir):
        return []

    available = []
    prefix = CONFIG["dataset"]["repo_prefix"]
    for entry in os.listdir(local_dir):
        entry_path = os.path.join(local_dir, entry)
        if os.path.isdir(entry_path):
            # Check if this is a valid dataset directory
            if os.path.exists(os.path.join(entry_path, "dataset_info.json")):
                # Skip old-format directories that don't use the encoding scheme
                # (directories created before encoding was implemented won't have _dash_ or _underscore_)
                if "_dash_" not in entry and "_underscore_" not in entry:
                    continue
                # Decode the directory name back to env_id
                env_id = _decode_hf_repo_name(entry)
                available.append(env_id)
    return sorted(set(available))


def _get_available_envs_from_hf():
    """Get list of env_ids that have HF Hub recordings."""
    try:
        user_info = whoami()
        username = (
            user_info.get("name") or user_info.get("user") or user_info.get("username")
        )
    except Exception:
        return []

    prefix = CONFIG["dataset"]["repo_prefix"]
    available = []

    try:
        api = HfApi()
        # List all datasets for this user
        datasets = api.list_datasets(author=username)
        for ds in datasets:
            if ds.id and ds.id.startswith(f"{username}/{prefix}"):
                env_id = hf_repo_id_to_env_id(ds.id)
                if env_id:
                    available.append(env_id)
    except Exception:
        pass

    return sorted(set(available))


def _get_metadata_path(env_id):
    """Return the path to the metadata JSON file for a given environment."""
    encoded_env_id = _encode_env_id_for_hf(env_id)
    return os.path.join(
        CONFIG["storage"]["local_dir"], f"{encoded_env_id}_metadata.json"
    )


def save_dataset_locally(dataset, env_id, metadata=None):
    """Save dataset to local disk, appending to any existing data."""
    path = get_local_dataset_path(env_id)
    metadata_path = _get_metadata_path(env_id)

    if os.path.exists(path):
        # Load existing dataset - UUIDs are already unique, no offsetting needed
        existing_dataset = load_from_disk(path)

        # Concatenate datasets
        from datasets import concatenate_datasets

        dataset = concatenate_datasets([existing_dataset, dataset])

        # Remove old dataset directory
        shutil.rmtree(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.save_to_disk(path)

    # Save/update metadata
    if metadata is not None:
        existing_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
        # Update with new metadata (newer values take precedence)
        existing_metadata.update(metadata)
        # Add recording timestamp
        if "recordings" not in existing_metadata:
            existing_metadata["recordings"] = []
        existing_metadata["recordings"].append(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "episodes": len(set(dataset["episode_id"])),
                "frames": len(dataset),
            }
        )
        with open(metadata_path, "w") as f:
            json.dump(existing_metadata, f, indent=2, default=_json_default)

    console.print(f"Dataset saved locally ([{STYLE_PATH}]{path}[/])")
    return path


def load_local_metadata(env_id):
    """Load metadata from local disk. Returns None if not found."""
    metadata_path = _get_metadata_path(env_id)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_local_dataset(env_id):
    """Load dataset from local disk. Returns None if not found."""
    path = get_local_dataset_path(env_id)
    if not os.path.exists(path):
        return None
    return load_from_disk(path)


def upload_local_dataset(env_id):
    """Load local dataset and push to Hugging Face Hub."""
    if not ensure_hf_login():
        return False
    dataset = load_local_dataset(env_id)
    if dataset is None:
        console.print(f"[{STYLE_FAIL}]No local dataset found for {env_id}[/]")
        console.print(
            f"  Expected at: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]"
        )
        return False
    hf_repo_id = env_id_to_hf_repo_id(env_id)
    try:
        dataset.push_to_hub(hf_repo_id)
        # Load metadata if available
        metadata = load_local_metadata(env_id)
        generate_dataset_card(dataset, env_id, hf_repo_id, metadata=metadata)
        console.print(
            f"[{STYLE_SUCCESS}]Dataset uploaded: https://huggingface.co/datasets/{hf_repo_id}[/]"
        )
        return True
    except Exception as e:
        console.print(f"[{STYLE_FAIL}]Upload failed: {e}[/]")
        console.print(
            f"To retry: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]"
        )
        return False


def minari_export(env_id, dataset_name=None, author=None):
    """Export a local HF dataset to Minari format for offline RL."""
    try:
        import minari
        from minari.data_collector import EpisodeBuffer
    except ImportError:
        console.print(f"[{STYLE_FAIL}]Minari is not installed.[/]")
        console.print(f"Install with: [{STYLE_CMD}]uv pip install 'minari>=0.5.0'[/]")
        return False

    dataset = load_local_dataset(env_id)
    if dataset is None:
        console.print(f"[{STYLE_FAIL}]No local dataset found for {env_id}[/]")
        console.print(
            f"  Expected at: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]"
        )
        return False

    # Group rows by episode
    episodes = {}
    for row in dataset:
        eid = row["episode_id"]
        # Handle both UUID bytes and legacy integer episode IDs
        if isinstance(eid, bytes):
            eid = uuid.UUID(bytes=eid)
        if eid not in episodes:
            episodes[eid] = []
        episodes[eid].append(row)
    for eid in episodes:
        if "step" in episodes[eid][0]:
            episodes[eid].sort(key=lambda r: r["step"])

    # Try to extract action/observation spaces from the environment
    action_space = None
    observation_space = None
    try:
        env = create_env(env_id)
        action_space = env.action_space
        observation_space = env.observation_space
        env.close()
    except Exception:
        console.print(
            f"[yellow]Could not create env for space metadata; inferring from data.[/]"
        )

    # Build EpisodeBuffers
    buffers = []
    total_steps = 0
    for ep_idx, (eid, rows) in enumerate(sorted(episodes.items())):
        observations = []
        actions = []
        rewards = []
        terminations = []
        truncations = []
        ep_seed = rows[0].get("seed", 0)

        for row in rows:
            img = row.get("observations", row.get("observation"))
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            observations.append(img)

            # Detect terminal observation by empty/missing actions
            action = row.get("actions", row.get("action"))
            if (isinstance(action, list) and len(action) == 0) or action is None:
                continue

            if isinstance(action, list) and len(action) == 1:
                action = action[0]
            actions.append(action)

            reward = row.get("rewards", row.get("reward"))
            rewards.append(float(reward) if reward is not None else 0.0)
            term = row.get("terminations", row.get("termination"))
            terminations.append(bool(term) if term is not None else False)
            trunc = row.get("truncations", row.get("truncation"))
            truncations.append(bool(trunc) if trunc is not None else False)

        # Fallback for old datasets without terminal observation:
        # duplicate last obs to satisfy Minari N+1 requirement
        if len(observations) == len(actions):
            observations.append(observations[-1])

        buffers.append(
            EpisodeBuffer(
                id=ep_idx,
                seed=ep_seed,
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
            )
        )
        total_steps += len(actions)

    env_id_underscored = env_id.replace("-", "_").replace("/", "_")
    if dataset_name is None:
        dataset_name = f"gymrec/{env_id_underscored}/human-v0"

    # Delete existing dataset with same name if present
    try:
        minari.delete_dataset(dataset_name)
    except Exception:
        pass

    create_kwargs = dict(
        dataset_id=dataset_name,
        buffer=buffers,
        algorithm_name="human",
        description=f"Human gameplay of {env_id}, recorded with gymrec",
    )
    if author:
        create_kwargs["author"] = author
    if action_space is not None:
        create_kwargs["action_space"] = action_space
    if observation_space is not None:
        create_kwargs["observation_space"] = observation_space

    minari.create_dataset_from_buffers(**create_kwargs)

    lines = [
        f"Episodes: [{STYLE_INFO}]{len(buffers)}[/]",
        f"Total steps: [{STYLE_INFO}]{total_steps}[/]",
        f"Dataset ID: [{STYLE_ENV}]{dataset_name}[/]",
        "",
        f"Load with:",
        f"  [{STYLE_CMD}]import minari[/]",
        f"  [{STYLE_CMD}]ds = minari.load_dataset('{dataset_name}')[/]",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title="Minari Export Complete",
            border_style="green",
            expand=False,
        )
    )
    return True


def _detect_backend(env_id):
    """Return backend name for metadata tags."""
    if env_id in set(_get_stableretro_envs()):
        return "stable-retro"
    elif "Vizdoom" in env_id:
        return "vizdoom"
    else:
        return "atari"


def _capture_env_metadata(env):
    """Capture environment configuration metadata for dataset card."""
    metadata = {
        "env_id": getattr(env, "_env_id", "unknown"),
        "backend": _detect_backend(getattr(env, "_env_id", "")),
        "frameskip": get_frameskip(env),
        "fps": get_default_fps(env),
    }

    # Action space info
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        metadata["action_space_type"] = "Discrete"
        metadata["n_actions"] = action_space.n
    elif isinstance(action_space, gym.spaces.MultiBinary):
        metadata["action_space_type"] = "MultiBinary"
        metadata["n_actions"] = action_space.n
    elif isinstance(action_space, gym.spaces.Dict):
        metadata["action_space_type"] = "Dict"
        for key, space in action_space.spaces.items():
            if isinstance(space, gym.spaces.Discrete):
                metadata[f"action_space_{key}"] = f"Discrete({space.n})"
            elif isinstance(space, gym.spaces.MultiBinary):
                metadata[f"action_space_{key}"] = f"MultiBinary({space.n})"
            else:
                metadata[f"action_space_{key}"] = str(space)
    else:
        metadata["action_space_type"] = type(action_space).__name__

    # Observation space shape
    obs_space = env.observation_space
    if hasattr(obs_space, "shape"):
        metadata["observation_shape"] = list(obs_space.shape)
    if hasattr(obs_space, "dtype"):
        metadata["observation_dtype"] = str(obs_space.dtype)

    # Spec-based metadata
    spec = getattr(env, "spec", None)
    if spec is not None:
        if hasattr(spec, "max_episode_steps") and spec.max_episode_steps is not None:
            metadata["max_episode_steps"] = spec.max_episode_steps
        kwargs = getattr(spec, "kwargs", {}) or {}
        # Sticky actions (ALE)
        if "repeat_action_probability" in kwargs:
            metadata["sticky_actions"] = kwargs["repeat_action_probability"]

    # ALE-specific: check unwrapped for sticky actions
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        sticky = getattr(unwrapped, "_repeat_action_probability", None)
        if sticky is not None:
            metadata["sticky_actions"] = sticky

    # Reward range
    if hasattr(env, "reward_range"):
        metadata["reward_range"] = list(env.reward_range)

    # Stable-Retro specific
    if hasattr(env, "_stable_retro") and env._stable_retro:
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            metadata["retro_platform"] = getattr(unwrapped, "system", None)
            metadata["retro_game"] = getattr(unwrapped, "gamerom", None)
            buttons = getattr(unwrapped, "buttons", None)
            if buttons:
                metadata["retro_buttons"] = list(buttons)

    # VizDoom specific
    if hasattr(env, "_vizdoom") and env._vizdoom:
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            metadata["vizdoom_scenario"] = getattr(unwrapped, "scenario", None)
            metadata["vizdoom_num_binary_buttons"] = getattr(
                unwrapped, "num_binary_buttons", None
            )
            metadata["vizdoom_num_delta_buttons"] = getattr(
                unwrapped, "num_delta_buttons", None
            )

    return metadata


def _size_category(n):
    """Return HF size_categories string for a frame count."""
    if n < 1000:
        return "n<1K"
    if n < 10000:
        return "1K<n<10K"
    if n < 100000:
        return "10K<n<100K"
    if n < 1000000:
        return "100K<n<1M"
    return "n>1M"


_BACKEND_LABELS = {
    "atari": "Atari (ALE-py)",
    "vizdoom": "VizDoom",
    "stable-retro": "Stable-Retro",
}


def generate_dataset_card(dataset, env_id, repo_id, metadata=None):
    """Generate or update the dataset card for a given dataset repo."""

    frames = len(dataset)
    episodes = len(set(dataset["episode_id"]))
    backend = _detect_backend(env_id)

    user_info = whoami()
    curator = user_info.get("name") or user_info.get("user") or "unknown"

    card_data = DatasetCardData(
        language="en",
        license=CONFIG["dataset"]["license"],
        task_categories=CONFIG["dataset"]["task_categories"],
        tags=["gymnasium", backend, env_id],
        size_categories=[_size_category(frames)],
        pretty_name=f"{env_id} Gameplay Dataset",
    )

    header = card_data.to_yaml()
    content_lines = [
        "---",
        header,
        "---",
        "",
        f"# {env_id} Gameplay Dataset",
        "",
        f"Human gameplay recordings from the Gymnasium environment `{env_id}`,",
        f"captured using [gymrec](https://github.com/tsilva/gymrec).",
        "",
        "## Dataset Summary",
        "",
        "| Stat | Value |",
        "|------|-------|",
        f"| Total frames | {frames:,} |",
        f"| Episodes | {episodes:,} |",
        f"| Environment | `{env_id}` |",
        f"| Backend | {_BACKEND_LABELS.get(backend, backend)} |",
        "",
    ]

    # Add Environment Configuration section if metadata is available
    if metadata:
        content_lines.extend(
            [
                "## Environment Configuration",
                "",
                "| Setting | Value |",
                "|---------|-------|",
            ]
        )

        # Core settings
        if "frameskip" in metadata:
            content_lines.append(f"| Frameskip | {metadata['frameskip']} |")
        if "fps" in metadata:
            content_lines.append(f"| Target FPS | {metadata['fps']} |")
        if "sticky_actions" in metadata:
            content_lines.append(f"| Sticky Actions | {metadata['sticky_actions']} |")
        if "max_episode_steps" in metadata:
            content_lines.append(
                f"| Max Episode Steps | {metadata['max_episode_steps']} |"
            )

        # Observation space
        if "observation_shape" in metadata:
            shape = metadata["observation_shape"]
            content_lines.append(
                f"| Observation Shape | {' × '.join(str(s) for s in shape)} |"
            )
        if "observation_dtype" in metadata:
            content_lines.append(
                f"| Observation Dtype | {metadata['observation_dtype']} |"
            )

        # Action space
        if "action_space_type" in metadata:
            content_lines.append(f"| Action Space | {metadata['action_space_type']} |")
        if "n_actions" in metadata:
            content_lines.append(f"| Number of Actions | {metadata['n_actions']} |")

        # Reward range
        if "reward_range" in metadata:
            rmin, rmax = metadata["reward_range"]
            content_lines.append(f"| Reward Range | [{rmin}, {rmax}] |")

        # Backend-specific info
        if backend == "stable-retro":
            if "retro_platform" in metadata and metadata["retro_platform"]:
                content_lines.append(f"| Platform | {metadata['retro_platform']} |")
            if "retro_game" in metadata and metadata["retro_game"]:
                content_lines.append(f"| Game | {metadata['retro_game']} |")
            if "retro_buttons" in metadata and metadata["retro_buttons"]:
                buttons = ", ".join(metadata["retro_buttons"][:8])
                if len(metadata["retro_buttons"]) > 8:
                    buttons += f" (+{len(metadata['retro_buttons']) - 8} more)"
                content_lines.append(f"| Buttons | {buttons} |")

        elif backend == "vizdoom":
            if "vizdoom_scenario" in metadata and metadata["vizdoom_scenario"]:
                content_lines.append(f"| Scenario | {metadata['vizdoom_scenario']} |")
            if "vizdoom_num_binary_buttons" in metadata:
                content_lines.append(
                    f"| Binary Buttons | {metadata['vizdoom_num_binary_buttons']} |"
                )
            if "vizdoom_num_delta_buttons" in metadata:
                content_lines.append(
                    f"| Delta Buttons | {metadata['vizdoom_num_delta_buttons']} |"
                )

        content_lines.append("")

    content_lines.extend(
        [
            "## Dataset Structure",
            "",
            "Minari-compatible flat table format. Use `minari-export` for native [Minari](https://minari.farama.org/) HDF5 format.",
            "",
            "Each episode has N step rows plus one terminal observation row (N+1 pattern).",
            "The terminal observation is the final state after the last step — it has an empty action",
            "and null values for rewards/terminations/truncations/infos.",
            "",
            "- **episode_id** (`binary(16)`): Unique UUID identifier for each episode (16 bytes, universally unique across all recordings)",
            "- **seed** (`int` or `null`): RNG seed used for `env.reset()` (set on first row of each episode, `null` on other rows)",
            "- **observations** (`Image`): RGB frame from the environment",
            "- **actions** (`list`): Action taken at this step (`[]` for terminal observations)",
            "- **rewards** (`float` or `null`): Reward received (`null` on terminal observation rows)",
            "- **terminations** (`bool` or `null`): Whether the episode terminated naturally (`null` on terminal observation rows)",
            "- **truncations** (`bool` or `null`): Whether the episode was truncated (`null` on terminal observation rows)",
            "- **infos** (`str` or `null`): Additional environment info as JSON (`null` on terminal observation rows)",
            "",
            "## Usage",
            "",
            "```python",
            "from datasets import load_dataset",
            f'ds = load_dataset("{repo_id}")',
            "```",
            "",
            "## About",
            "",
            "Recorded with [gymrec](https://github.com/tsilva/gymrec).",
            f"Curated by: {curator}",
        ]
    )
    card = DatasetCard("\n".join(content_lines))

    card.push_to_hub(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=CONFIG["dataset"]["commit_message"],
    )


def _create_env__stableretro(env_id):
    import stable_retro as retro

    try:
        env = retro.make(env_id, render_mode="rgb_array")
    except FileNotFoundError:
        console.print(f"\n[{STYLE_FAIL}]Error: ROM not found for '{env_id}'.[/]")
        console.print(f"\nStable-retro requires ROM files to be imported separately.")
        console.print(
            f"Import ROMs with:  [{STYLE_CMD}]python -m stable_retro.import /path/to/your/roms/[/]"
        )
        console.print(
            f"\nUse [{STYLE_CMD}]list_environments[/] to see which games have ROMs imported."
        )
        sys.exit(1)
    env._stable_retro = True
    return env


def _create_env__vizdoom(env_id):
    from vizdoom import gymnasium_wrapper

    env = gym.make(env_id, render_mode="rgb_array", max_buttons_pressed=0)
    env._vizdoom = True
    return env


def _create_env__alepy(env_id):
    import ale_py

    gym.register_envs(ale_py)
    return gym.make(env_id, render_mode="rgb_array")


def create_env(env_id):
    """Create a Gymnasium environment with the appropriate backend."""

    if env_id in set(_get_stableretro_envs()):
        env = _create_env__stableretro(env_id)
    elif "Vizdoom" in env_id:
        env = _create_env__vizdoom(env_id)
    else:
        env = _create_env__alepy(env_id)

    env._env_id = env_id.replace("-", "_")
    return env


def get_frameskip(env) -> int:
    """Detect the frameskip value for an environment.

    Returns the number of internal frames per env.step() call.
    For stochastic frameskip tuples like (2, 5), returns the average.
    """
    env_id = getattr(env, "_env_id", "")

    # VizDoom and Retro have different frameskip semantics; skip detection
    if hasattr(env, "_vizdoom") and env._vizdoom:
        return 1
    if hasattr(env, "_stable_retro") and env._stable_retro:
        return 1

    # Check spec kwargs (works for gym.make() environments)
    spec = getattr(env, "spec", None)
    if spec is not None:
        kwargs = getattr(spec, "kwargs", {}) or {}
        fs = kwargs.get("frameskip")
        if fs is not None:
            if isinstance(fs, (tuple, list)) and len(fs) == 2:
                return max(int(round((fs[0] + fs[1]) / 2)), 1)
            try:
                return max(int(fs), 1)
            except (TypeError, ValueError):
                pass

    # ALE-specific attribute fallback
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        fs = getattr(unwrapped, "_frameskip", None)
        if fs is not None:
            if isinstance(fs, (tuple, list)) and len(fs) == 2:
                return max(int(round((fs[0] + fs[1]) / 2)), 1)
            try:
                return max(int(fs), 1)
            except (TypeError, ValueError):
                pass

    # Name-based: NoFrameskip means frameskip=1
    if "NoFrameskip" in env_id:
        return 1

    return 1


def get_default_fps(env):
    """Determine a sensible default FPS for an environment."""

    base_fps = None

    for key in ("render_fps", "video.frames_per_second"):
        fps = env.metadata.get(key)
        if fps:
            try:
                base_fps = int(round(float(fps)))
                break
            except (TypeError, ValueError):
                pass

    if base_fps is None:
        env_id = getattr(env, "_env_id", "")

        retro_platforms = {
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

        if (
            any(env_id.endswith(f"-{plat}") for plat in retro_platforms)
            or "Atari2600" in env_id
        ):
            base_fps = CONFIG["fps_defaults"]["retro"]
        elif "Vizdoom" in env_id or "vizdoom" in env_id:
            base_fps = CONFIG["fps_defaults"]["vizdoom"]
        else:
            base_fps = CONFIG["fps_defaults"]["atari"]

    frameskip = get_frameskip(env)
    if frameskip > 1:
        adjusted_fps = max(int(round(base_fps / frameskip)), 1)
        console.print(
            f"[{STYLE_INFO}]\\[FPS][/] Base FPS={base_fps}, frameskip={frameskip} → adjusted to [{STYLE_INFO}]{adjusted_fps}[/] FPS for human playability"
        )
        return adjusted_fps

    return base_fps


def _get_atari_envs() -> list[str]:
    try:
        import ale_py

        gym.register_envs(ale_py)
        return sorted(
            env_id
            for env_id in gym.envs.registry.keys()
            if str(gym.spec(env_id).entry_point) == "ale_py.env:AtariEnv"
            and env_id.startswith("ALE/")
        )
    except Exception:
        return []


def _get_stableretro_envs(imported_only: bool = False) -> list[str]:
    try:
        import stable_retro as retro

        all_games = sorted(retro.data.list_games(retro.data.Integrations.ALL))
        if imported_only:
            result = []
            for game in all_games:
                try:
                    retro.data.get_romfile_path(game, retro.data.Integrations.ALL)
                    result.append(game)
                except FileNotFoundError:
                    pass
            return result
        return all_games
    except Exception:
        return []


def _get_vizdoom_envs() -> list[str]:
    try:
        import vizdoom
        import vizdoom.gymnasium_wrapper

        return sorted(
            env_id
            for env_id in gym.envs.registry.keys()
            if env_id.startswith("Vizdoom")
        )
    except Exception:
        return []


def _get_env_platform(env_id: str) -> str:
    """Determine the platform type for an env_id."""
    retro_platforms = {
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
        "Atari2600",
    }

    if env_id.startswith("ALE/"):
        return "Atari"
    elif env_id.startswith("Vizdoom") or env_id.startswith("vizdoom"):
        return "VizDoom"
    elif any(env_id.endswith(f"-{plat}") for plat in retro_platforms):
        return "Stable-Retro"
    else:
        return "Atari"  # Default fallback


def select_environment_interactive(available_recordings_only: bool = False) -> str:
    from simple_term_menu import TerminalMenu

    if available_recordings_only:
        # Get envs with available recordings
        local_envs = _get_available_envs_from_local()
        hf_envs = _get_available_envs_from_hf()
        all_recorded_envs = sorted(set(local_envs + hf_envs))

        if not all_recorded_envs:
            console.print(
                f"[{STYLE_FAIL}]No recordings found.[/]\n"
                f"  Local path: [{STYLE_PATH}]{CONFIG['storage']['local_dir']}[/]\n"
                f"  Record first: [{STYLE_CMD}]uv run python main.py record <env_id>[/]"
            )
            raise SystemExit(1)

        # Group by platform
        atari_envs = [e for e in all_recorded_envs if _get_env_platform(e) == "Atari"]
        retro_envs = [
            e for e in all_recorded_envs if _get_env_platform(e) == "Stable-Retro"
        ]
        vizdoom_envs = [
            e for e in all_recorded_envs if _get_env_platform(e) == "VizDoom"
        ]

        entries = []
        env_id_map = []

        for env_id in atari_envs:
            entries.append(f"[Atari]  {env_id}")
            env_id_map.append(env_id)
        if atari_envs and (retro_envs or vizdoom_envs):
            entries.append("")
            env_id_map.append(None)

        for env_id in retro_envs:
            entries.append(f"[Stable-Retro]  {env_id}")
            env_id_map.append(env_id)
        if retro_envs and vizdoom_envs:
            entries.append("")
            env_id_map.append(None)

        for env_id in vizdoom_envs:
            entries.append(f"[VizDoom]  {env_id}")
            env_id_map.append(env_id)

        title = "  Select Recording\n"
        status_bar = "  ↑↓ navigate · / search · Enter select · Esc cancel"
    else:
        # Original behavior: list all available environments
        atari_envs = _get_atari_envs()
        retro_envs = _get_stableretro_envs(imported_only=True)
        vizdoom_envs = _get_vizdoom_envs()

        entries = []
        env_id_map = []

        for env_id in atari_envs:
            entries.append(f"[Atari]  {env_id}")
            env_id_map.append(env_id)
        if atari_envs and (retro_envs or vizdoom_envs):
            entries.append("")
            env_id_map.append(None)

        for env_id in retro_envs:
            entries.append(f"[Stable-Retro]  {env_id}")
            env_id_map.append(env_id)
        if retro_envs and vizdoom_envs:
            entries.append("")
            env_id_map.append(None)

        for env_id in vizdoom_envs:
            entries.append(f"[VizDoom]  {env_id}")
            env_id_map.append(env_id)

        if not entries:
            console.print(
                "[dim]No environments found. Install ale-py, stable-retro, or vizdoom.[/]"
            )
            raise SystemExit(1)

        title = "  Select Environment\n"
        status_bar = "  ↑↓ navigate · / search · Enter select · Esc cancel"

    menu = TerminalMenu(
        entries,
        title=title,
        menu_cursor="  > ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("bold", "fg_cyan"),
        search_highlight_style=("fg_black", "bg_cyan", "bold"),
        show_search_hint=True,
        show_search_hint_text="  (type / to search)",
        search_key="/",
        skip_empty_entries=True,
        status_bar=status_bar,
        status_bar_style=("fg_gray",),
    )

    selected_index = menu.show()
    if selected_index is None:
        console.print("[dim]No environment selected.[/]")
        raise SystemExit(0)

    return env_id_map[selected_index]


def _list_environments__alepy():
    atari_ids = _get_atari_envs()
    if atari_ids:
        lines = "\n".join(atari_ids)
    else:
        lines = "[dim]Could not list Atari environments.[/]"
    console.print(
        Panel(
            lines,
            title="[bold]Atari Environments[/]",
            border_style=STYLE_INFO,
            expand=False,
        )
    )


def _list_environments__stableretro():
    all_games = _get_stableretro_envs()
    if all_games:
        lines = []
        try:
            import stable_retro as retro

            for game in all_games:
                try:
                    retro.data.get_romfile_path(game, retro.data.Integrations.ALL)
                    lines.append(f"{game} [{STYLE_SUCCESS}](imported)[/]")
                except FileNotFoundError:
                    lines.append(f"{game} [{STYLE_FAIL}](missing ROM)[/]")
        except Exception as e:
            lines.append(f"[{STYLE_FAIL}]Stable-Retro not installed: {e}[/]")
        console.print(
            Panel(
                "\n".join(lines),
                title="[bold]Stable-Retro Games[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "[dim]Stable-Retro not installed or no games found.[/]",
                title="[bold]Stable-Retro Games[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )


def _list_environments__vizdoom():
    vizdoom_ids = _get_vizdoom_envs()
    if vizdoom_ids:
        lines = list(vizdoom_ids)
        console.print(
            Panel(
                "\n".join(lines),
                title="[bold]VizDoom Environments[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "[dim]VizDoom not installed.[/]",
                title="[bold]VizDoom Environments[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )


def list_environments():
    """Print available Atari, stable-retro and VizDoom environments."""
    _list_environments__alepy()
    _list_environments__stableretro()
    _list_environments__vizdoom()


def _import_roms(path: str):
    """Import ROMs into stable-retro from a directory or file."""
    import io
    import zipfile
    import stable_retro.data

    if not os.path.exists(path):
        console.print(f"[{STYLE_FAIL}]Error: Path not found: {path}[/]")
        return

    known_hashes = stable_retro.data.get_known_hashes()
    imported_games = 0

    def save_if_matches(filename, f):
        nonlocal imported_games
        try:
            data, hash = stable_retro.data.groom_rom(filename, f)
        except (OSError, ValueError):
            return
        if hash in known_hashes:
            game, ext, curpath = known_hashes[hash]
            game_path = os.path.join(curpath, game)
            rom_path = os.path.join(game_path, "rom%s" % ext)
            with open(rom_path, "wb") as f:
                f.write(data)

            metadata_path = os.path.join(game_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as mf:
                        metadata = json.load(mf)
                    original_name = metadata.get("original_rom_name")
                    if original_name:
                        with open(os.path.join(game_path, original_name), "wb") as of:
                            of.write(data)
                except (json.JSONDecodeError, OSError):
                    pass
            imported_games += 1

    def check_zipfile(f, process_f):
        with zipfile.ZipFile(f) as zf:
            for entry in zf.infolist():
                _root, ext = os.path.splitext(entry.filename)
                with zf.open(entry) as innerf:
                    if ext == ".zip":
                        check_zipfile(innerf, process_f)
                    else:
                        process_f(entry.filename, innerf)

    if os.path.isfile(path):
        # Single file
        with open(path, "rb") as f:
            _root, ext = os.path.splitext(path)
            if ext == ".zip":
                save_if_matches(os.path.basename(path), f)
                f.seek(0)
                try:
                    check_zipfile(f, save_if_matches)
                except zipfile.BadZipFile:
                    pass
            else:
                save_if_matches(os.path.basename(path), f)
    else:
        # Directory - walk recursively
        for root, dirs, files in os.walk(path):
            for filename in files:
                filepath = os.path.join(root, filename)
                with open(filepath, "rb") as f:
                    _root, ext = os.path.splitext(filename)
                    if ext == ".zip":
                        save_if_matches(filename, f)
                        f.seek(0)
                        try:
                            check_zipfile(f, save_if_matches)
                        except zipfile.BadZipFile:
                            pass
                    else:
                        save_if_matches(filename, f)

    console.print(f"[{STYLE_SUCCESS}]Imported {imported_games} ROM(s)[/]")


async def main():
    parser = argparse.ArgumentParser(description="Gymnasium Recorder/Playback")
    subparsers = parser.add_subparsers(dest="command")

    parser_record = subparsers.add_parser("record", help="Record gameplay")
    parser_record.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )
    parser_record.add_argument(
        "--fps", type=int, default=None, help="Frames per second for playback/recording"
    )
    parser_record.add_argument(
        "--scale", type=int, default=None, help="Display scale factor (default: 2)"
    )
    parser_record.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Record without uploading to Hugging Face (no HF account required)",
    )

    parser_playback = subparsers.add_parser("playback", help="Replay a dataset")
    parser_playback.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )
    parser_playback.add_argument(
        "--fps", type=int, default=None, help="Frames per second for playback/recording"
    )
    parser_playback.add_argument(
        "--scale", type=int, default=None, help="Display scale factor (default: 2)"
    )
    parser_playback.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify determinism by comparing replayed frames against recorded frames (pixel MSE)",
    )

    parser_upload = subparsers.add_parser(
        "upload", help="Upload local dataset to Hugging Face Hub"
    )
    parser_upload.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )

    subparsers.add_parser("login", help="Log in to Hugging Face Hub")
    subparsers.add_parser("list_environments", help="List available environments")

    parser_import = subparsers.add_parser(
        "import_roms", help="Import ROMs into stable-retro from a directory or file"
    )
    parser_import.add_argument(
        "path",
        type=str,
        help="Path to directory or file containing ROMs",
    )

    parser_minari = subparsers.add_parser(
        "minari-export", help="Export local dataset to Minari format"
    )
    parser_minari.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )
    parser_minari.add_argument(
        "--name",
        type=str,
        default=None,
        help="Minari dataset name (default: gymrec/<env_id>/human-v0)",
    )
    parser_minari.add_argument(
        "--author", type=str, default=None, help="Author name for dataset metadata"
    )

    args = parser.parse_args()
    if args.command is None:
        args.command = "record"
        for attr, default in [
            ("env_id", None),
            ("fps", None),
            ("scale", None),
            ("dry_run", False),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)

    if args.command == "login":
        _lazy_init()
        ensure_hf_login(force=True)
        return

    if args.command == "list_environments":
        list_environments()
        return

    if args.command == "import_roms":
        _import_roms(args.path)
        return

    env_id = args.env_id

    _lazy_init()

    if env_id is None:
        # For playback, only show environments with available recordings
        is_playback = args.command == "playback"
        env_id = select_environment_interactive(available_recordings_only=is_playback)

    if args.command == "upload":
        upload_local_dataset(env_id)
        return

    if args.command == "minari-export":
        minari_export(env_id, dataset_name=args.name, author=args.author)
        return

    if hasattr(args, "scale") and args.scale is not None:
        CONFIG["display"]["scale_factor"] = args.scale

    env = create_env(env_id)
    fps = args.fps if args.fps is not None else get_default_fps(env)

    if args.command == "record":
        recorder = DatasetRecorderWrapper(env)
        recorded_dataset = await recorder.record(fps=fps)

        if recorded_dataset is None:
            recorder.close()
            return

        save_dataset_locally(recorded_dataset, env_id, metadata=recorder._env_metadata)
        recorder.close()  # cleanup temp files after dataset is saved
        console.print(
            f"To play back: [{STYLE_CMD}]uv run python main.py playback {env_id}[/]"
        )

        if not args.dry_run:
            try:
                do_upload = Confirm.ask(
                    "Upload to Hugging Face Hub?", default=True, console=console
                )
            except EOFError:
                do_upload = False
            if do_upload:
                if not upload_local_dataset(env_id):
                    console.print(
                        f"To retry: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]"
                    )
            else:
                console.print(
                    f"To upload later: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]"
                )
    elif args.command == "playback":
        loaded_dataset = load_local_dataset(env_id)
        if loaded_dataset is not None:
            console.print(
                f"[{STYLE_INFO}]Playing back from local dataset ({len(loaded_dataset)} frames)[/]"
            )
            total = len(loaded_dataset)
        else:
            console.print("[dim]No local dataset found, trying Hugging Face Hub...[/]")
            try:
                hf_repo_id = env_id_to_hf_repo_id(env_id)
                api = HfApi()
                api.dataset_info(hf_repo_id)
                loaded_dataset = load_dataset(hf_repo_id, split="train", streaming=True)
            except Exception:
                loaded_dataset = None
            if loaded_dataset is None:
                console.print(f"[{STYLE_FAIL}]No dataset found for {env_id}.[/]")
                console.print(
                    f"  Local path: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]"
                )
                console.print(
                    f"  Record a session first: [{STYLE_CMD}]uv run python main.py record {env_id}[/]"
                )
                return
            try:
                builder = load_dataset_builder(hf_repo_id)
                if builder.info.splits and "train" in builder.info.splits:
                    total = builder.info.splits["train"].num_examples
                else:
                    total = None
            except Exception:
                total = None
            console.print(
                f"[{STYLE_INFO}]Playing back streaming from Hugging Face Hub[/]"
            )
        recorder = DatasetRecorderWrapper(env)

        def _is_step_row(row):
            """Filter out terminal observation rows."""
            action = row.get("actions", row.get("action"))
            if action is None or (isinstance(action, list) and len(action) == 0):
                return False
            return True

        if args.verify:
            recorded_data = (
                (
                    row.get("actions", row.get("action")),
                    row.get("observations", row.get("observation")),
                    row.get("rewards", row.get("reward")),
                    row.get("terminations", row.get("termination")),
                    row.get("truncations", row.get("truncation")),
                )
                for row in loaded_dataset
                if _is_step_row(row)
            )
            await recorder.replay(recorded_data, fps=fps, total=total, verify=True)
        else:
            actions = (
                row.get("actions", row.get("action"))
                for row in loaded_dataset
                if _is_step_row(row)
            )
            await recorder.replay(actions, fps=fps, total=total)


def cli():
    asyncio.run(main())


if __name__ == "__main__":
    cli()
