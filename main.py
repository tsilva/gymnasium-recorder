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

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

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
            pygame.K_z: 0, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_x: 8,
        },
        "Atari2600": {
            pygame.K_z: 0, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
        },
        "Snes": {
            pygame.K_z: 0, pygame.K_a: 1, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_x: 8, pygame.K_s: 9,
            pygame.K_q: 10, pygame.K_w: 11,
        },
        "GbAdvance": {
            pygame.K_z: 0, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_x: 8, pygame.K_a: 10,
            pygame.K_s: 11,
        },
        "GameBoy": {
            pygame.K_z: 0, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_x: 8,
        },
        "GbColor": {
            pygame.K_z: 0, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_x: 8,
        },
        "PCEngine": {
            pygame.K_x: 0, pygame.K_c: 1, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_z: 8, pygame.K_a: 9,
            pygame.K_s: 10, pygame.K_d: 11,
        },
        "Saturn": {
            pygame.K_x: 0, pygame.K_z: 1, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_c: 8, pygame.K_a: 9,
            pygame.K_s: 10, pygame.K_d: 11,
        },
        "32x": {
            pygame.K_x: 0, pygame.K_z: 1, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_c: 8, pygame.K_a: 9,
            pygame.K_s: 10, pygame.K_d: 11,
        },
        "Genesis": {
            pygame.K_x: 0, pygame.K_z: 1, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_c: 8, pygame.K_a: 9,
            pygame.K_s: 10, pygame.K_d: 11,
        },
        "Sms": {
            pygame.K_z: 0, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_x: 8,
        },
        "GameGear": {
            pygame.K_z: 0, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_x: 8,
        },
        "SCD": {
            pygame.K_x: 0, pygame.K_z: 1, pygame.K_n: 2, pygame.K_m: 3,
            pygame.K_UP: 4, pygame.K_DOWN: 5, pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7, pygame.K_c: 8, pygame.K_a: 9,
            pygame.K_s: 10, pygame.K_d: 11,
        },
    }

    # Try loading config file
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keymappings.toml")
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
    "recording": {"jpeg_quality": 95},
    "fps_defaults": {"atari": 90, "vizdoom": 45, "retro": 90},
    "dataset": {
        "repo_prefix": "GymnasiumRecording__",
        "license": "mit",
        "task_categories": ["reinforcement-learning"],
        "commit_message": "Update dataset card",
    },
    "storage": {
        "local_dir": os.path.join(os.path.expanduser("~"), ".gymnasium-recorder", "datasets"),
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
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
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
    global whoami, DatasetCard, DatasetCardData, HfApi
    global Dataset, Value, Sequence, HFImage, load_dataset, load_from_disk, load_dataset_builder, concatenate_datasets
    global START_KEY, ATARI_KEY_BINDINGS, VIZDOOM_KEY_BINDINGS, STABLE_RETRO_KEY_BINDINGS

    import numpy as np
    import pygame
    from PIL import Image as PILImage
    from huggingface_hub import whoami, DatasetCard, DatasetCardData, HfApi
    from datasets import (
        Dataset,
        Value,
        Sequence,
        Image as HFImage,
        load_dataset,
        load_from_disk,
        load_dataset_builder,
        concatenate_datasets,
    )

    START_KEY, ATARI_KEY_BINDINGS, VIZDOOM_KEY_BINDINGS, STABLE_RETRO_KEY_BINDINGS = _load_keymappings(pygame)
    CONFIG = _load_config()

class DatasetRecorderWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for recording and replaying Atari gameplay as Hugging Face datasets.
    """
    def __init__(self, env):
        _lazy_init()
        super().__init__(env)

        self.recording = False
        self.frame_shape = None  # Delay initialization
        self.screen = None       # Delay initialization

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
        self.frames = []
        self.actions = []
        self.steps = []
        self.timestamps = []
        self.rewards = []
        self.terminateds = []
        self.truncateds = []
        self.infos = []

        self.temp_dir = tempfile.mkdtemp()

        self._fps = None
        self._fps_changed_at = 0
        self._episode_count = 0
        self._cumulative_reward = 0.0
        self._overlay_visible = True
        self._recorded_dataset = None

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
            self.screen = pygame.display.set_mode((self.frame_shape[1] * scale, self.frame_shape[0] * scale))
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
        path = os.path.join(self.temp_dir, f"frame_{len(self.frames):05d}.jpg")
        PILImage.fromarray(frame_uint8).save(path, quality=CONFIG["recording"]["jpeg_quality"])
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

    def _resolve_atari_key_mapping(self):
        """Resolve meaning-based Atari key bindings to action indices using the env's action meanings."""
        # Standard fallback: meaning -> index for the default Atari action order
        standard_meaning_to_idx = {
            "NOOP": 0, "FIRE": 1, "UP": 2, "RIGHT": 3, "LEFT": 4, "DOWN": 5,
            "UPRIGHT": 6, "UPLEFT": 7, "DOWNRIGHT": 8, "DOWNLEFT": 9,
            "UPFIRE": 10, "RIGHTFIRE": 11, "LEFTFIRE": 12, "DOWNFIRE": 13,
            "UPRIGHTFIRE": 14, "UPLEFTFIRE": 15, "DOWNRIGHTFIRE": 16, "DOWNLEFTFIRE": 17,
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
            if isinstance(value, int):
                # Legacy: direct index mapping (from old config files)
                resolved[key] = value
            elif isinstance(value, str):
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
            if hasattr(self.env, '_vizdoom') and self.env._vizdoom:
                return self._get_user_action__vizdoom()
            if hasattr(self.env, '_stable_retro') and self.env._stable_retro:
                return self._get_user_action__stableretro()
            return self._get_user_action__alepy()

    def _get_user_action__vizdoom(self):
        return self._get_vizdoom_action()

    def _get_user_action__stableretro(self):
        return self._get_stable_retro_action()

    def _get_user_action__alepy(self):
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

        if hasattr(self.env, '_vizdoom') and self.env._vizdoom:
            env_type = "VizDoom"
            if self._vizdoom_buttons is None:
                self._vizdoom_buttons = self._init_vizdoom_key_mapping()
            for key, action_name in VIZDOOM_KEY_BINDINGS.items():
                btn_idx = self._vizdoom_buttons.get(action_name)
                idx_str = f"btn {btn_idx}" if btn_idx is not None else ""
                table.add_row(pygame.key.name(key), action_name, idx_str)
            ml_idx = self._vizdoom_buttons.get("MOVE_LEFT")
            mr_idx = self._vizdoom_buttons.get("MOVE_RIGHT")
            table.add_row("alt+left", "MOVE_LEFT", f"btn {ml_idx}" if ml_idx is not None else "")
            table.add_row("alt+right", "MOVE_RIGHT", f"btn {mr_idx}" if mr_idx is not None else "")
        elif hasattr(self.env, '_stable_retro') and self.env._stable_retro:
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
                label = buttons[idx] if buttons and idx < len(buttons) else f"button {idx}"
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
                label = meanings[action_idx] if meanings and action_idx < len(meanings) else f"action {action_idx}"
                table.add_row(pygame.key.name(key), label, f"action {action_idx}")

        table.add_section()
        table.add_row("[dim]escape[/]", "[dim]Exit[/]", "")
        table.add_row("[dim]+/-[/]", "[dim]Adjust FPS (±5)[/]", "")

        console.print(Panel(table, title=f"[{STYLE_ENV}]{env_type}[/] Key Mappings", border_style=STYLE_INFO, expand=False))

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
            clock.tick(overlay_cfg["fps"])

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
            await self._play(fps)  # bypass play() to avoid premature close()
            return self._recorded_dataset
        except:
            self.close()  # cleanup on error only
            raise
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
        self._fps = fps

        self.episode_ids.clear()
        self.frames.clear()
        self.actions.clear()
        self.steps.clear()
        self.timestamps.clear()
        self.rewards.clear()
        self.terminateds.clear()
        self.truncateds.clear()
        self.infos.clear()

        episode_id = int(time.time())
        self._episode_count = 1
        self._cumulative_reward = 0.0
        obs, _ = self.env.reset()
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
            if not self._input_loop(): break
            action = self._get_user_action()
            self._record_frame(episode_id, step, obs, action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._cumulative_reward += float(reward)
            if self.recording:
                self.rewards.append(float(reward))
                self.terminateds.append(bool(terminated))
                self.truncateds.append(bool(truncated))
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
                obs, _ = self.env.reset()
                self._episode_count += 1
                self._cumulative_reward = 0.0
                episode_id = int(time.time())
                step = 0
                self._render_frame(obs)

        if self.recording and self.frames:
            data = {
                "episode_id": self.episode_ids,
                "timestamp": self.timestamps,
                "image": self.frames,
                "step": self.steps,
                "action": self.actions,
                "reward": self.rewards,
                "terminated": self.terminateds,
                "truncated": self.truncateds,
                "info": self.infos,
            }
            self._recorded_dataset = Dataset.from_dict(data)
            self._recorded_dataset = self._recorded_dataset.cast_column("image", HFImage())

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
            if isinstance(self.env.action_space, gym.spaces.Discrete) and len(action) == 1:
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
                if not self._input_loop(): break

                if verify:
                    action, recorded_image, recorded_reward, recorded_terminated, recorded_truncated = item
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
                        mse = float(np.mean((obs_image.astype(np.float32) - recorded_array.astype(np.float32)) ** 2))
                    else:
                        console.print(f"  [yellow]Warning:[/] shape mismatch at frame {len(verify_metrics)}: "
                              f"obs={obs_image.shape} vs recorded={recorded_array.shape}, skipping comparison")
                        mse = None

                    if float(reward) != float(recorded_reward):
                        reward_mismatches += 1
                    if bool(terminated) != bool(recorded_terminated) or bool(truncated) != bool(recorded_truncated):
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
                lines.append(f"Frame MSE:  mean=[{STYLE_INFO}]{mean_mse:.2f}[/], max=[{STYLE_INFO}]{max_mse:.2f}[/], min=[{STYLE_INFO}]{min_mse:.2f}[/]")
                lines.append(f"Reward mismatches: [{STYLE_INFO}]{reward_mismatches}/{n_total}[/] frames")
                lines.append(f"Terminal state mismatches: [{STYLE_INFO}]{terminal_mismatches}/{n_total}[/] frames")

                passed = exceeded == 0 and reward_mismatches == 0 and terminal_mismatches == 0
                if passed:
                    lines.append(f"Result: [{STYLE_SUCCESS}]PASS[/] (all frames below threshold {mse_threshold})")
                    border_style = "green"
                else:
                    reasons = []
                    if exceeded > 0:
                        reasons.append(f"{exceeded} frames exceeded MSE threshold {mse_threshold}")
                    if reward_mismatches > 0:
                        reasons.append(f"{reward_mismatches} reward mismatches")
                    if terminal_mismatches > 0:
                        reasons.append(f"{terminal_mismatches} terminal state mismatches")
                    lines.append(f"Result: [{STYLE_FAIL}]FAIL[/] ({', '.join(reasons)})")
                    border_style = "red"
            else:
                lines.append("[yellow]No valid frame comparisons (all skipped).[/]")
                border_style = "yellow"

            console.print()
            console.print(Panel("\n".join(lines), title="Determinism Verification Report", border_style=border_style, expand=False))

    def close(self):
        """
        Clean up resources and save dataset if needed.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        pygame.quit()
        super().close()

def env_id_to_hf_repo_id(env_id):
    user_info = whoami()
    username = user_info.get("name") or user_info.get("user") or user_info.get("username")
    env_id_underscored = env_id.replace("-", "_").replace("/", "_")
    hf_repo_id = f"{username}/{CONFIG['dataset']['repo_prefix']}{env_id_underscored}"
    return hf_repo_id


def get_local_dataset_path(env_id):
    """Return the local storage path for a given environment's dataset."""
    env_id_underscored = env_id.replace("-", "_").replace("/", "_")
    return os.path.join(CONFIG["storage"]["local_dir"], env_id_underscored)


def save_dataset_locally(dataset, env_id):
    """Save dataset to local disk, concatenating with existing data if present."""
    path = get_local_dataset_path(env_id)
    existing = load_local_dataset(env_id)
    if existing is not None:
        dataset = concatenate_datasets([existing, dataset])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.save_to_disk(path)
    console.print(f"Dataset saved locally ([{STYLE_PATH}]{path}[/])")
    return path


def load_local_dataset(env_id):
    """Load dataset from local disk. Returns None if not found."""
    path = get_local_dataset_path(env_id)
    if not os.path.exists(path):
        return None
    return load_from_disk(path)


def filter_dataset_by_episode(dataset, episode_arg):
    """Filter dataset by episode selection. Returns (filtered_dataset, episode_label)."""
    episode_ids = sorted(set(dataset["episode_id"]))
    if len(episode_ids) <= 1:
        return dataset, None

    if episode_arg == "all":
        return dataset, f"all {len(episode_ids)} episodes"

    if episode_arg == "latest":
        target = episode_ids[-1]
        idx = len(episode_ids)
    else:
        try:
            n = int(episode_arg)
            if n < 1 or n > len(episode_ids):
                console.print(f"[yellow]Episode {n} out of range (1-{len(episode_ids)}). Using latest.[/]")
                target = episode_ids[-1]
                idx = len(episode_ids)
            else:
                target = episode_ids[n - 1]
                idx = n
        except ValueError:
            console.print(f"[yellow]Invalid episode '{episode_arg}'. Using latest.[/]")
            target = episode_ids[-1]
            idx = len(episode_ids)

    filtered = dataset.filter(lambda row: row["episode_id"] == target)
    return filtered, f"episode {idx}/{len(episode_ids)}"


def upload_local_dataset(env_id):
    """Load local dataset and push to Hugging Face Hub."""
    dataset = load_local_dataset(env_id)
    if dataset is None:
        console.print(f"[{STYLE_FAIL}]No local dataset found for {env_id}[/]")
        console.print(f"  Expected at: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]")
        return False
    hf_repo_id = env_id_to_hf_repo_id(env_id)
    try:
        dataset.push_to_hub(hf_repo_id)
        generate_dataset_card(dataset, env_id, hf_repo_id)
        console.print(f"[{STYLE_SUCCESS}]Dataset uploaded to {hf_repo_id}[/]")
        return True
    except Exception as e:
        console.print(f"[{STYLE_FAIL}]Upload failed: {e}[/]")
        console.print(f"To retry: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]")
        return False


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
        license=CONFIG["dataset"]["license"],
        task_categories=CONFIG["dataset"]["task_categories"],
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
        commit_message=CONFIG["dataset"]["commit_message"],
    )

def _create_env__stableretro(env_id):
    import stable_retro as retro
    try:
        env = retro.make(env_id, render_mode="rgb_array")
    except FileNotFoundError:
        console.print(f"\n[{STYLE_FAIL}]Error: ROM not found for '{env_id}'.[/]")
        console.print(f"\nStable-retro requires ROM files to be imported separately.")
        console.print(f"Import ROMs with:  [{STYLE_CMD}]python -m stable_retro.import /path/to/your/roms/[/]")
        console.print(f"\nUse [{STYLE_CMD}]list_environments[/] to see which games have ROMs imported.")
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

def _get_default_fps__stableretro(env_id: str) -> int:
    """Return a default FPS for stable-retro environments."""
    return CONFIG["fps_defaults"]["retro"]


def _get_default_fps__vizdoom(env_id: str) -> int:
    """Return a default FPS for VizDoom environments."""
    return CONFIG["fps_defaults"]["vizdoom"]


def _get_default_fps__alepy(env_id: str) -> int:
    """Return a default FPS for Atari environments."""
    return CONFIG["fps_defaults"]["atari"]


def get_frameskip(env) -> int:
    """Detect the frameskip value for an environment.

    Returns the number of internal frames per env.step() call.
    For stochastic frameskip tuples like (2, 5), returns the average.
    """
    env_id = getattr(env, "_env_id", "")

    # VizDoom and Retro have different frameskip semantics; skip detection
    if hasattr(env, '_vizdoom') and env._vizdoom:
        return 1
    if hasattr(env, '_stable_retro') and env._stable_retro:
        return 1

    # Check spec kwargs (works for gym.make() environments)
    spec = getattr(env, 'spec', None)
    if spec is not None:
        kwargs = getattr(spec, 'kwargs', {}) or {}
        fs = kwargs.get('frameskip')
        if fs is not None:
            if isinstance(fs, (tuple, list)) and len(fs) == 2:
                return max(int(round((fs[0] + fs[1]) / 2)), 1)
            try:
                return max(int(fs), 1)
            except (TypeError, ValueError):
                pass

    # ALE-specific attribute fallback
    unwrapped = getattr(env, 'unwrapped', None)
    if unwrapped is not None:
        fs = getattr(unwrapped, '_frameskip', None)
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
            base_fps = _get_default_fps__stableretro(env_id)
        elif "Vizdoom" in env_id or "vizdoom" in env_id:
            base_fps = _get_default_fps__vizdoom(env_id)
        else:
            base_fps = _get_default_fps__alepy(env_id)

    frameskip = get_frameskip(env)
    if frameskip > 1:
        adjusted_fps = max(int(round(base_fps / frameskip)), 1)
        console.print(f"[{STYLE_INFO}]\\[FPS][/] Base FPS={base_fps}, frameskip={frameskip} → adjusted to [{STYLE_INFO}]{adjusted_fps}[/] FPS for human playability")
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
            env_id for env_id in gym.envs.registry.keys() if env_id.startswith("Vizdoom")
        )
    except Exception:
        return []


def select_environment_interactive() -> str:
    from simple_term_menu import TerminalMenu

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
        console.print("[dim]No environments found. Install ale-py, stable-retro, or vizdoom.[/]")
        raise SystemExit(1)

    menu = TerminalMenu(
        entries,
        title="  Select Environment\n",
        menu_cursor="  > ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("bold", "fg_cyan"),
        search_highlight_style=("fg_black", "bg_cyan", "bold"),
        show_search_hint=True,
        show_search_hint_text="  (type / to search)",
        search_key="/",
        skip_empty_entries=True,
        status_bar="  ↑↓ navigate · / search · Enter select · Esc cancel",
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
    console.print(Panel(lines, title="[bold]Atari Environments[/]", border_style=STYLE_INFO, expand=False))


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
        console.print(Panel("\n".join(lines), title="[bold]Stable-Retro Games[/]", border_style=STYLE_INFO, expand=False))
    else:
        console.print(Panel("[dim]Stable-Retro not installed or no games found.[/]", title="[bold]Stable-Retro Games[/]", border_style=STYLE_INFO, expand=False))


def _list_environments__vizdoom():
    vizdoom_ids = _get_vizdoom_envs()
    if vizdoom_ids:
        lines = list(vizdoom_ids)
        try:
            import vizdoom
            if getattr(vizdoom, "wads", None):
                lines.append("")
                lines.append("[bold]Available WADs:[/]")
                for wad in vizdoom.wads:
                    lines.append(f"  {wad}")
        except Exception:
            pass
        console.print(Panel("\n".join(lines), title="[bold]VizDoom Environments[/]", border_style=STYLE_INFO, expand=False))
    else:
        console.print(Panel("[dim]VizDoom not installed.[/]", title="[bold]VizDoom Environments[/]", border_style=STYLE_INFO, expand=False))


def list_environments():
    """Print available Atari, stable-retro and VizDoom environments."""
    _list_environments__alepy()
    _list_environments__stableretro()
    _list_environments__vizdoom()

async def main():
    parser = argparse.ArgumentParser(description="Atari Gymnasium Recorder/Playback")
    subparsers = parser.add_subparsers(dest="command")

    parser_record = subparsers.add_parser("record", help="Record gameplay")
    parser_record.add_argument("env_id", type=str, nargs="?", default=None, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")
    parser_record.add_argument("--fps", type=int, default=None, help="Frames per second for playback/recording")
    parser_record.add_argument("--scale", type=int, default=None, help="Display scale factor (default: 2)")
    parser_record.add_argument("--jpeg-quality", type=int, default=None, help="JPEG recording quality 1-100 (default: 95)")
    parser_record.add_argument("--dry-run", action="store_true", default=False,
        help="Record without uploading to Hugging Face (no HF account required)")

    parser_playback = subparsers.add_parser("playback", help="Replay a dataset")
    parser_playback.add_argument("env_id", type=str, nargs="?", default=None, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")
    parser_playback.add_argument("--fps", type=int, default=None, help="Frames per second for playback/recording")
    parser_playback.add_argument("--scale", type=int, default=None, help="Display scale factor (default: 2)")
    parser_playback.add_argument("--episode", type=str, default="latest",
        help="Episode to play: 'latest' (default), 'all', or episode number (1-based)")
    parser_playback.add_argument("--verify", action="store_true", default=False,
        help="Verify determinism by comparing replayed frames against recorded frames (pixel MSE)")

    parser_upload = subparsers.add_parser("upload", help="Upload local dataset to Hugging Face Hub")
    parser_upload.add_argument("env_id", type=str, nargs="?", default=None, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")

    subparsers.add_parser("list_environments", help="List available environments")

    args = parser.parse_args()
    if args.command is None:
        args.command = "record"
        for attr, default in [("env_id", None), ("fps", None), ("scale", None), ("jpeg_quality", None), ("dry_run", False)]:
            if not hasattr(args, attr):
                setattr(args, attr, default)

    if args.command == "list_environments":
        list_environments()
        return

    env_id = args.env_id
    if env_id is None:
        env_id = select_environment_interactive()

    _lazy_init()

    if args.command == "upload":
        upload_local_dataset(env_id)
        return

    if hasattr(args, 'scale') and args.scale is not None:
        CONFIG["display"]["scale_factor"] = args.scale
    if hasattr(args, 'jpeg_quality') and args.jpeg_quality is not None:
        CONFIG["recording"]["jpeg_quality"] = args.jpeg_quality

    env = create_env(env_id)
    fps = args.fps if args.fps is not None else get_default_fps(env)

    if args.command == "record":
        recorder = DatasetRecorderWrapper(env)
        recorded_dataset = await recorder.record(fps=fps)

        save_dataset_locally(recorded_dataset, env_id)
        recorder.close()  # cleanup temp files after dataset is saved
        console.print(f"To play back: [{STYLE_CMD}]uv run python main.py playback {env_id}[/]")

        if not args.dry_run:
            try:
                do_upload = Confirm.ask("Upload to Hugging Face Hub?", default=True, console=console)
            except EOFError:
                do_upload = False
            if do_upload:
                if not upload_local_dataset(env_id):
                    console.print(f"To retry: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]")
            else:
                console.print(f"To upload later: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]")
    elif args.command == "playback":
        loaded_dataset = load_local_dataset(env_id)
        if loaded_dataset is not None:
            loaded_dataset, ep_label = filter_dataset_by_episode(loaded_dataset, args.episode)
            info = f"{len(loaded_dataset)} frames"
            if ep_label:
                info += f", {ep_label}"
            console.print(f"[{STYLE_INFO}]Playing back from local dataset ({info})[/]")
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
                console.print(f"  Local path: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]")
                console.print(f"  Record a session first: [{STYLE_CMD}]uv run python main.py record {env_id}[/]")
                return
            try:
                builder = load_dataset_builder(hf_repo_id)
                if builder.info.splits and "train" in builder.info.splits:
                    total = builder.info.splits["train"].num_examples
                else:
                    total = None
            except Exception:
                total = None
        recorder = DatasetRecorderWrapper(env)
        if args.verify:
            recorded_data = (
                (row["action"], row["image"], row["reward"], row["terminated"], row["truncated"])
                for row in loaded_dataset
            )
            await recorder.replay(recorded_data, fps=fps, total=total, verify=True)
        else:
            actions = (row["action"] for row in loaded_dataset)
            await recorder.replay(actions, fps=fps, total=total)

if __name__ == "__main__":
    asyncio.run(main())
