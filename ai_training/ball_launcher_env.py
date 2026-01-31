import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class BallLauncherEnv(gym.Env):
    """
    A port of the Ball Launcher game physics to a Gymnasium environment.
    Logical size: 800x600.
    """
    def __init__(self, num_targets=3, render_mode=None):
        super(BallLauncherEnv, self).__init__()
        
        self.logical_width = 800
        self.logical_height = 600
        self.gravity = 900
        self.ball_radius = 10
        self.min_velocity = 100
        self.max_velocity = 1500
        self.elasticity = 0.8
        self.dt = 1/60
        self.substeps = 5
        self.max_steps = 500
        
        self.num_targets = num_targets
        self.render_mode = render_mode
        
        # Action space: [velocity, angle (in radians)]
        # Angle is between 0.1 and PI-0.1 (aiming upwards)
        self.action_space = spaces.Box(
            low=np.array([self.min_velocity, 0.1], dtype=np.float32),
            high=np.array([self.max_velocity, math.pi - 0.1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: Target positions (x, y, r) for each target
        # For simplicity in CEM, we might not even use the observation if we optimize for a fixed config.
        # But let's define it for future NN use.
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.logical_width, self.logical_height),
            shape=(num_targets * 3,),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None, target_configs=None):
        super().reset(seed=seed)
        
        if target_configs:
            self.targets = []
            for cfg in target_configs:
                self.targets.append({
                    'x': cfg['x'],
                    'y': cfg['y'],
                    'r': cfg['r'],
                    'hit': False,
                    'min_dist': float('inf')
                })
        else:
            # Generate random targets similar to JS logic
            self.targets = []
            for _ in range(self.num_targets):
                self.targets.append({
                    'x': 100 + np.random.rand() * (self.logical_width - 200),
                    'y': 100 + np.random.rand() * (self.logical_height - 200),
                    'r': 15 + np.random.rand() * 25,
                    'hit': False,
                    'min_dist': float('inf')
                })
        
        self.ball = {
            'x': self.logical_width / 2,
            'y': self.logical_height - 20,
            'vx': 0,
            'vy': 0
        }
        
        self.launched = False
        self.done = False
        self.step_count = 0
        self.total_reward = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for t in self.targets:
            obs.extend([t['x'], t['y'], t['r']])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}
        
        velocity, angle = action
        self.ball['vx'] = velocity * math.cos(angle)
        self.ball['vy'] = -velocity * math.sin(angle)
        self.launched = True
        
        hit_reward = 0
        
        # Simulate until termination
        while not self.done:
            for _ in range(self.substeps):
                dt = self.dt / self.substeps
                
                # Update Ball
                self.ball['vy'] += self.gravity * dt
                self.ball['x'] += self.ball['vx'] * dt
                self.ball['y'] += self.ball['vy'] * dt
                
                # Check Targets
                for t in self.targets:
                    dx = self.ball['x'] - t['x']
                    dy = self.ball['y'] - t['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if not t['hit'] and dist < t['min_dist']:
                        t['min_dist'] = dist
                        
                    if dist < (self.ball_radius + t['r']):
                        if not t['hit']:
                            t['hit'] = True
                            hit_reward += 100
                            
                        # Physics: Resolve Collision
                        nx = dx / dist
                        ny = dy / dist
                        v_dot_n = self.ball['vx'] * nx + self.ball['vy'] * ny
                        
                        if v_dot_n < 0:
                            j = -(1 + self.elasticity) * v_dot_n
                            self.ball['vx'] += j * nx
                            self.ball['vy'] += j * ny
                            
                            overlap = (self.ball_radius + t['r']) - dist
                            self.ball['x'] += nx * overlap
                            self.ball['y'] += ny * overlap
            
            self.step_count += 1
            self.total_reward += hit_reward
            hit_reward = 0 # reset for next step if we were doing step-by-step
            
            # Termination logic
            all_hit = all(t['hit'] for t in self.targets)
            out_of_bounds = (
                self.ball['x'] < -200 or 
                self.ball['x'] > self.logical_width + 200 or 
                self.ball['y'] > self.logical_height + 50
            )
            
            if all_hit or out_of_bounds or self.step_count >= self.max_steps:
                self.done = True
                if all_hit:
                    self.total_reward += 200
                
                # Shaped reward
                for t in self.targets:
                    if not t['hit']:
                        # 20 * exp(-dist/50)
                        r = 20.0 * math.exp(-t['min_dist'] / 50.0)
                        self.total_reward += r
                        
        return self._get_obs(), self.total_reward, self.done, False, {}

