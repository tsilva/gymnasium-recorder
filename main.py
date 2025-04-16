import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from datetime import datetime
import os
import pygame
import threading
import asyncio
import pandas as pd
import sys
import tempfile
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image as HFImage, load_dataset
from huggingface_hub import login, whoami
from dotenv import load_dotenv

class AtariDatasetRecorder(gym.Wrapper):
    def __init__(self, env, output_dir="datasets", record=True, hf_repo_id=None):
        super().__init__(env)
        self.output_dir = output_dir
        self.record = record
        self.frame_shape = self.observation_space.shape
        self.frame_counter = 0
        self.episode_seed = None
        self.recording_started = False
        self.hf_repo_id = hf_repo_id

        self.env_id = self.env.spec.id.replace("-", "_")
        os.makedirs(self.output_dir, exist_ok=True)

        pygame.init()
        self.screen = pygame.display.set_mode((self.frame_shape[1], self.frame_shape[0]))
        pygame.display.set_caption(f"Atari Game - {self.env_id}")

        self.current_keys = set()
        self.key_lock = threading.Lock()

        self.key_to_action = {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_LEFT: 3,
        }
        self.default_action = 0

        # For dataset
        self.frames = []
        self.actions = []
        self.temp_dir = tempfile.mkdtemp()

    def reset(self, **kwargs):
        self.recording_started = False
        self.frame_counter = 0
        self.frames = []
        self.actions = []

        if self.record:
            if self.episode_seed is None:
                self.episode_seed = kwargs.get('seed', int(datetime.now().timestamp()))
        else:
            self.episode_seed = getattr(self, 'episode_seed', None)

        if self.episode_seed is not None:
            kwargs['seed'] = self.episode_seed

        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.record:
            if not self.recording_started and action != self.default_action:
                self.recording_started = True
            if self.recording_started:
                self._record_frame(obs, action)

        if terminated or truncated:
            if self.record and self.recording_started:
                self._save_dataset()

        return obs, reward, terminated, truncated, info

    def _record_frame(self, frame, action):
        # Save frame as PNG to temp dir, store path
        img = PILImage.fromarray(frame.astype(np.uint8))
        path = os.path.join(self.temp_dir, f"frame_{self.frame_counter:05d}.png")
        img.save(path)
        self.frames.append(path)
        self.actions.append(int(action))
        self.frame_counter += 1

    def _save_dataset(self):
        # Save as Hugging Face dataset
        features = Features({
            "image": HFImage(),
            "action": Value("int64")
        })
        data = {
            "image": self.frames,
            "action": self.actions
        }
        ds = Dataset.from_dict(data, features=features)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = os.path.join(self.output_dir, f"{self.env_id}_episode_{timestamp}")
        ds.save_to_disk(local_path)
        print(f"Saved dataset to {local_path}")

        # Try to upload to HF Hub if repo_id and token available
        if self.hf_repo_id:
            try:
                print(f"Pushing to Hugging Face Hub: {self.hf_repo_id}")
                ds.push_to_hub(self.hf_repo_id)
                print("Upload complete.")
            except Exception as e:
                print(f"Failed to upload to Hugging Face Hub: {e}")

    def _input_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                with self.key_lock:
                    self.current_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                with self.key_lock:
                    self.current_keys.discard(event.key)
        return True

    def _get_action(self):
        with self.key_lock:
            for key in self.current_keys:
                if key in self.key_to_action:
                    return self.key_to_action[key]
        return self.default_action

    def _render_frame(self, frame):
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    async def play(self):
        obs, info = self.reset()
        done = False
        clock = pygame.time.Clock()

        while not done:
            if not self._input_loop():
                break
            action = self._get_action()
            obs, reward, terminated, truncated, info = self.step(action)
            self._render_frame(obs)
            done = terminated or truncated
            clock.tick(30)
            await asyncio.sleep(1.0 / 30)

    def replay_from_hf(self, dataset_path):
        # dataset_path can be a local path or a HF repo id
        ds = load_dataset(dataset_path, split="train")
        print(f"Loaded dataset with {len(ds)} frames")
        clock = pygame.time.Clock()
        for example in ds:
            img = np.array(example["image"])
            self._render_frame(img)
            clock.tick(30)

    def close(self):
        # Save dataset if recording started and not yet saved
        if self.record and self.recording_started and self.frames:
            self._save_dataset()
        pygame.quit()
        super().close()

async def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = None
    if hf_token:
        login(token=hf_token)
        try:
            user_info = whoami()
            username = user_info.get("name") or user_info.get("user") or user_info.get("username")
        except Exception as e:
            print(f"Could not determine Hugging Face username: {e}")
            username = None
    else:
        username = None

    if len(sys.argv) < 2:
        print("Usage: python record.py <env_id> [hf_dataset_or_local_path]")
        print("Example: python record.py BreakoutNoFrameskip-v4")
        print("         python record.py BreakoutNoFrameskip-v4 username/dataset_name")
        sys.exit(1)

    env_id = sys.argv[1]
    if "NoFrameskip" not in env_id:
        print("Error: Only NoFrameskip environments are supported.")
        sys.exit(1)

    env_id_underscored = env_id.replace("-", "_")
    if username:
        hf_repo_id = f"{username}/{env_id_underscored}"

    if len(sys.argv) > 2:
        # Replay from HF dataset or local dataset
        env = gym.make(env_id, render_mode="rgb_array")
        recorder = AtariDatasetRecorder(env, output_dir="datasets", record=False)
        try:
            recorder.replay_from_hf(sys.argv[2])
        finally:
            recorder.close()
    else:
        # Record new gameplay as HF dataset
        env = gym.make(env_id, render_mode="rgb_array")
        recorder = AtariDatasetRecorder(env, output_dir="datasets", record=True, hf_repo_id=hf_repo_id)
        try:
            await recorder.play()
        finally:
            recorder.close()

if __name__ == "__main__":
    asyncio.run(main())
