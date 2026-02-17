# TODO

## High Priority

- [ ] Store rewards, done flags, and info in dataset (reward, terminated, truncated, info columns)
- [ ] Clean up temp directory after recording (use `tempfile.TemporaryDirectory` or `shutil.rmtree`)
- [ ] Multi-episode recording sessions (auto-reset on game over, keep recording until ESC)

## Medium Priority

- [ ] Actual determinism verification on playback (frame-diff comparison via pixel MSE or hash)
- [ ] Agent recording / programmatic API (record RL agent rollouts, not just human keyboard input)
- [ ] Dataset metadata for reproducibility (action/observation space, env kwargs, library versions)
- [ ] Episode statistics and filtering (`stats` command, drop short episodes, action distributions)

## Low Priority

- [ ] Export to Minari / D4RL format for offline RL library compatibility
- [ ] Add Doom WAD support
- [ ] Add validation for dataset fields
