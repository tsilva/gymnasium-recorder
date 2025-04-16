import os
import sys
import gymnasium as gym
import ale_py
import numpy as np
import pygame
import threading
import asyncio
import tempfile
from datetime import datetime
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image as HFImage, load_dataset
from huggingface_hub import login, whoami
from dotenv import load_dotenv
import argparse

gym.register_envs(ale_py)

class AtariDatasetRecorder(gym.Wrapper):
    """
    Gymnasium wrapper for recording and replaying Atari gameplay as Hugging Face datasets.
    """
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
        self.frames = []
        self.actions = []
        self.temp_dir = tempfile.mkdtemp()

    def reset(self, **kwargs):
        """
        Reset the environment and prepare for recording.
        """
        self.recording_started = False
        self.frame_counter = 0
        self.frames.clear()
        self.actions.clear()
        if self.record:
            if self.episode_seed is None:
                self.episode_seed = kwargs.get('seed', int(datetime.now().timestamp()))
        else:
            self.episode_seed = getattr(self, 'episode_seed', None)
        if self.episode_seed is not None:
            kwargs['seed'] = self.episode_seed
        obs, info = self.env.reset(**kwargs)
        if self.record:
            self._record_frame(obs, self.default_action)
        return obs, info

    def step(self, action):
        """
        Take a step in the environment and record the frame/action if enabled.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.record:
            self._record_frame(obs, action)
        if (terminated or truncated) and self.record and self.recording_started:
            self._save_dataset()
        return obs, reward, terminated, truncated, info

    def _record_frame(self, frame, action):
        """
        Save a frame and action to temporary storage.
        """
        img = PILImage.fromarray(frame.astype(np.uint8))
        path = os.path.join(self.temp_dir, f"frame_{self.frame_counter:05d}.png")
        img.save(path, format="PNG")
        self.frames.append(path)
        self.actions.append(int(action))
        self.frame_counter += 1
        self.recording_started = True

    def _save_dataset(self):
        """
        Save the recorded frames and actions as a Hugging Face dataset.
        """
        features = Features({
            "image": HFImage(),
            "action": Value("int64")
        })
        data = {"image": self.frames, "action": self.actions}
        ds = Dataset.from_dict(data, features=features)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = os.path.join(self.output_dir, f"{self.env_id}_episode_{timestamp}")
        ds.save_to_disk(local_path)
        if self.hf_repo_id:
            try:
                ds.push_to_hub(self.hf_repo_id)
            except Exception as e:
                print(f"Failed to upload to Hugging Face Hub: {e}")

    def _input_loop(self):
        """
        Handle pygame input events.
        """
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
        """
        Map pressed keys to actions.
        """
        with self.key_lock:
            for key in self.current_keys:
                if key in self.key_to_action:
                    return self.key_to_action[key]
        return self.default_action

    def _render_frame(self, frame):
        """
        Render a frame using pygame.
        """
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    async def play(self):
        """
        Main loop for interactive gameplay and recording.
        """
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
        """
        Replay a recorded dataset from Hugging Face or local disk.
        """
        ds = load_dataset(dataset_path, split="train")
        clock = pygame.time.Clock()
        obs, info = self.env.reset()
        img0 = np.array(ds[0]["image"])
        if hasattr(ds[0]["image"], "mode") and ds[0]["image"].mode == "P":
            img0 = ds[0]["image"].convert("RGB")
            img0 = np.array(img0)
        if not np.array_equal(obs, img0):
            print("Warning: Initial frame mismatch at step 0")
            PILImage.fromarray(obs).save("frame_env_0.png")
            PILImage.fromarray(img0).save("frame_dataset_0.png")
            sys.exit(1)
        self._render_frame(img0)
        prev_action = int(ds[0]["action"])
        for i in range(1, len(ds)):
            img = np.array(ds[i]["image"])
            if hasattr(ds[i]["image"], "mode") and ds[i]["image"].mode == "P":
                img = ds[i]["image"].convert("RGB")
                img = np.array(img)
            action = int(ds[i]["action"])
            obs, reward, terminated, truncated, info = self.env.step(prev_action)
            if not np.array_equal(obs, img):
                print(f"Warning: Frame mismatch at step {i}")
                PILImage.fromarray(obs).save(f"frame_env_{i}.png")
                PILImage.fromarray(img).save(f"frame_dataset_{i}.png")
                sys.exit(1)
            self._render_frame(img)
            prev_action = action
            clock.tick(30)

    def close(self):
        """
        Clean up resources and save dataset if needed.
        """
        if self.record and self.recording_started and self.frames:
            self._save_dataset()
        pygame.quit()
        super().close()

async def main():
    """
    Entry point for CLI: record or playback mode.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Atari Gymnasium Recorder/Playback")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    record_parser = subparsers.add_parser("record", help="Record a new gameplay session")
    record_parser.add_argument("env_id", type=str, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")
    playback_parser = subparsers.add_parser("playback", help="Replay a recorded dataset")
    playback_parser.add_argument("dataset_path", type=str, help="HF dataset repo id or local path")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = None
    username = None
    if hf_token:
        login(token=hf_token)
        try:
            user_info = whoami()
            username = user_info.get("name") or user_info.get("user") or user_info.get("username")
        except Exception:
            pass

    if args.mode == "record":
        env_id = args.env_id
        if "NoFrameskip" not in env_id:
            print("Error: Only NoFrameskip environments are supported.")
            sys.exit(1)
        env_id_underscored = env_id.replace("-", "_")
        if username:
            hf_repo_id = f"{username}/{env_id_underscored}"
        env = gym.make(env_id, render_mode="rgb_array")
        recorder = AtariDatasetRecorder(env, output_dir="datasets", record=True, hf_repo_id=hf_repo_id)
        try:
            await recorder.play()
        finally:
            recorder.close()
    elif args.mode == "playback":
        dataset_path = args.dataset_path
        env_id = "BreakoutNoFrameskip-v4"
        if "/" in dataset_path:
            env_id_guess = dataset_path.split("/")[-1].replace("_", "-")
            if "NoFrameskip" in env_id_guess:
                env_id = env_id_guess
            repo_id = dataset_path
        else:
            env_id = dataset_path
            env_id_underscored = env_id.replace("-", "_")
            if username:
                repo_id = f"{username}/{env_id_underscored}"
            else:
                print("Error: Could not determine Hugging Face username for dataset playback.")
                sys.exit(1)
        env = gym.make(env_id, render_mode="rgb_array")
        recorder = AtariDatasetRecorder(env, output_dir="datasets", record=False)
        try:
            recorder.replay_from_hf(repo_id)
        finally:
            recorder.close()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
