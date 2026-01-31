# Ball Launcher Game

An interactive physics-based ball launcher game built with HTML5 Canvas and JavaScript. This is the playable web component of a larger reinforcement learning project that uses Gymnasium and PyTorch to train agents to master the game.

## About

Launch a ball to hit all targets on screen by adjusting velocity and angle. The game features realistic physics including gravity, collisions, and elastic bouncing off targets.

## Features

- **Adjustable Launch Parameters**: Control velocity (↑/↓) and angle (←/→)
- **Fine Control Mode**: Hold SHIFT for precise adjustments
- **Configurable Targets**: Choose 1-10 targets via dropdown or URL parameter (`?targets=5`)
- **Score System**: Earn points for hitting targets, with bonus for hitting all targets
- **Audio Feedback**: Musical tones play when targets are hit

## How to Play

1. Open `index.html` in a web browser
2. Use arrow keys to adjust velocity and angle
3. Press SPACE to launch the ball
4. Press R to retry with the same target configuration
5. Press ESC to generate new random targets

## Controls

- **↑/↓**: Adjust velocity (100-1500)
- **←/→**: Adjust angle (0-180°)
- **SHIFT**: Fine adjustment mode
- **SPACE**: Launch ball
- **R**: Retry same configuration
- **ESC**: Generate new targets

## Configuration

Set the number of targets via URL parameter:
```
index.html?targets=5
```

Or use the dropdown menu in the game interface.

