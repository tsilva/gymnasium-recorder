
import os
import time
import numpy as np
import pygame
import threading
import asyncio
import tempfile
from PIL import Image as PILImage
from datasets import Dataset, Features, Value, Image as HFImage, load_dataset, concatenate_datasets
from huggingface_hub import whoami
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

REPO_PREFIX = "GymnasiumRecording__"

class AtariDatasetRecorder(gym.Wrapper):
    """
    Gymnasium wrapper for recording and replaying Atari gameplay as Hugging Face datasets.
    """
    def __init__(self, env):
        super().__init__(env)

        self.recording = False
        self.frame_shape = self.observation_space.shape
        self.env_id = self.env.spec.id.replace("-", "_")

        pygame.init()
        self.screen = pygame.display.set_mode((self.frame_shape[1], self.frame_shape[0]))
        pygame.display.set_caption(self.env_id)

        self.current_keys = set()
        self.key_lock = threading.Lock()
        self.key_to_action = {
            pygame.K_UP: 1,
            pygame.K_RIGHT: 2,
            pygame.K_LEFT: 3,
            pygame.K_DOWN: 4
        }
        self.noop_action = 0

        self.episode_ids = []
        self.frames = []
        self.actions = []
        self.steps = []

        self.temp_dir = tempfile.mkdtemp()

    def _record_frame(self, episode_id, step, frame, action):
        """
        Save a frame and action to temporary storage.
        """
        if not self.recording: return
        img = PILImage.fromarray(frame.astype(np.uint8))
        path = os.path.join(self.temp_dir, f"frame_{len(self.frames):05d}.png")
        img.save(path, format="PNG")
        self.episode_ids.append(episode_id)
        self.steps.append(step)
        self.frames.append(path)
        self.actions.append(int(action))

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
        Map pressed keys to actions.
        """
        with self.key_lock:
            for key in self.current_keys:
                if key in self.key_to_action:
                    return self.key_to_action[key]
        return self.noop_action

    def _render_frame(self, frame):
        """
        Render a frame using pygame.
        """
        if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    async def record(self, fps=30):
        self.recording = True
        try: return await self._record(fps=fps)
        finally: self.recording = False

    async def _record(self, fps=30):
        self.recording = True
        try: 
            await self.play(fps=fps)
            features = Features({"episode_id": Value("int64"), "image": HFImage(), "step": Value("int64") ,"action": Value("int64")})
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

async def main():
    parser = argparse.ArgumentParser(description="Atari Gymnasium Recorder/Playback")
    parser.add_argument("mode", type=str, choices=["record", "playback"], help="Mode of operation: 'record' or 'playback'")
    parser.add_argument("env_id", type=str, help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for playback")
    args = parser.parse_args()

    env_id = args.env_id
    hf_repo_id = env_id_to_hf_repo_id(env_id)
    try: loaded_dataset = load_dataset(hf_repo_id, split="train")
    except: pass

    if args.mode == "record":
        env = gym.make(env_id, render_mode="rgb_array")
        recorder = AtariDatasetRecorder(env)
        recorded_dataset = await recorder.record(fps=args.fps)
        final_dataset = concatenate_datasets([loaded_dataset, recorded_dataset]) if loaded_dataset else recorded_dataset
        final_dataset.push_to_hub(hf_repo_id)
    elif args.mode == "playback":
        assert loaded_dataset is not None, f"Dataset not found: {hf_repo_id}"
        env = gym.make(env_id, render_mode="rgb_array")
        recorder = AtariDatasetRecorder(env)
        actions = loaded_dataset["action"]
        await recorder.replay(actions, fps=args.fps)

if __name__ == "__main__":
    asyncio.run(main())
