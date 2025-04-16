import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import cv2
import numpy as np
from datetime import datetime
import os
import pygame
import threading
import queue
import asyncio

class AtariVideoRecorder(gym.Wrapper):
    def __init__(self, env, output_dir="videos", fps=30):
        super().__init__(env)
        self.output_dir = output_dir
        self.fps = fps
        self.frame_shape = self.observation_space.shape
        self.episode_count = 0
        self.frame_counter = 0
        self.action_log = []

        # Use the environment's ID as the base name
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

        self.frame_queue = queue.Queue()
        self.video_writer = None
        self.video_filename = None
        self.writer_thread = None
        self.stop_writer = threading.Event()

    def reset(self, **kwargs):
        self._stop_video_writer()

        self.episode_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_filename = f"{self.env_id}_episode_{self.episode_count}_{timestamp}"
        self.video_filename = os.path.join(self.output_dir, base_filename + ".avi")
        self.log_filename = os.path.join(self.output_dir, base_filename + "_actions.csv")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            self.video_filename, fourcc, self.fps,
            (self.frame_shape[1], self.frame_shape[0])
        )

        self.stop_writer.clear()
        self.writer_thread = threading.Thread(target=self._video_writer_loop)
        self.writer_thread.start()

        self.frame_counter = 0
        self.action_log = []

        obs, info = self.env.reset(**kwargs)
        self._queue_frame(obs, action=None)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._queue_frame(obs, action)

        if terminated or truncated:
            self._stop_video_writer()

        return obs, reward, terminated, truncated, info

    def _queue_frame(self, frame, action):
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.frame_queue.put((frame_bgr, action))
        self.action_log.append({
            'frame': self.frame_counter,
            'action': action
        })
        self.frame_counter += 1

    def _video_writer_loop(self):
        while not self.stop_writer.is_set() or not self.frame_queue.empty():
            try:
                frame_bgr, _ = self.frame_queue.get(timeout=0.1)
                self.video_writer.write(frame_bgr)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
        self.video_writer.release()

    def _stop_video_writer(self):
        if self.writer_thread is not None:
            self.stop_writer.set()
            self.writer_thread.join()
            self.writer_thread = None

            # Save actions to CSV
            if self.log_filename:
                with open(self.log_filename, 'w') as f:
                    f.write("frame,action\n")
                    for entry in self.action_log:
                        f.write(f"{entry['frame']},{entry['action']}\n")

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
            clock.tick(self.fps)
            await asyncio.sleep(1.0 / self.fps)

    def close(self):
        self._stop_video_writer()
        pygame.quit()
        super().close()

async def main():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = AtariVideoRecorder(env, output_dir="videos", fps=30)
    try: await env.play()
    finally: env.close()

if __name__ == "__main__":
    asyncio.run(main())