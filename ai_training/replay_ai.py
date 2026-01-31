import json
import os
import math
import pygame
import sys
from ball_launcher_env import BallLauncherEnv

# Constants for rendering (match JS logic)
LOGICAL_WIDTH = 800
LOGICAL_HEIGHT = 600

def replay():
    if not os.path.exists('results/best_solution.json'):
        print("No solution found. Run train_ai.py first.")
        return
    
    with open('results/best_solution.json', 'r') as f:
        data = json.load(f)
        
    target_configs = data['target_configs']
    best_action = data['best_action']
    best_reward = data['best_reward']
    
    # Initialize Environment
    env = BallLauncherEnv(num_targets=len(target_configs))
    env.reset(target_configs=target_configs)
    
    # Action setup
    velocity, angle = best_action
    env.ball['vx'] = velocity * math.cos(angle)
    env.ball['vy'] = -velocity * math.sin(angle)
    env.launched = True
    
    # Pygame Setup
    pygame.init()
    screen = pygame.display.set_mode((LOGICAL_WIDTH, LOGICAL_HEIGHT))
    pygame.display.set_caption("Ball Launcher AI Replay")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("Arial", 24)
    
    trajectory = []
    done = False
    
    print("--- Visualizing AI Solution ---")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Restart
                    env.reset(target_configs=target_configs)
                    env.ball['vx'] = velocity * math.cos(angle)
                    env.ball['vy'] = -velocity * math.sin(angle)
                    env.launched = True
                    trajectory = []
                    done = False

        if not done:
            # Step-by-step physics for smoother animation
            for _ in range(env.substeps):
                dt = env.dt / env.substeps
                
                # Update Ball
                env.ball['vy'] += env.gravity * dt
                env.ball['x'] += env.ball['vx'] * dt
                env.ball['y'] += env.ball['vy'] * dt
                
                # Check Targets
                for t in env.targets:
                    dx = env.ball['x'] - t['x']
                    dy = env.ball['y'] - t['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < (env.ball_radius + t['r']):
                        if not t['hit']:
                            t['hit'] = True
                            
                        # Physics: Resolve Collision
                        nx = dx / dist
                        ny = dy / dist
                        v_dot_n = env.ball['vx'] * nx + env.ball['vy'] * ny
                        
                        if v_dot_n < 0:
                            j = -(1 + env.elasticity) * v_dot_n
                            env.ball['vx'] += j * nx
                            env.ball['vy'] += j * ny
                            
                            overlap = (env.ball_radius + t['r']) - dist
                            env.ball['x'] += nx * overlap
                            env.ball['y'] += ny * overlap
            
            trajectory.append((env.ball['x'], env.ball['y']))
            env.step_count += 1
            
            # Termination logic
            all_hit = all(t['hit'] for t in env.targets)
            out_of_bounds = (
                env.ball['x'] < -200 or 
                env.ball['x'] > env.logical_width + 200 or 
                env.ball['y'] > env.logical_height + 50
            )
            
            if all_hit or out_of_bounds or env.step_count >= env.max_steps:
                done = True

        # Render
        screen.fill((30, 30, 30)) # background
        
        # Draw Trajectory
        if len(trajectory) > 1:
            pygame.draw.lines(screen, (100, 100, 100), False, trajectory, 2)
            
        # Draw Targets
        for t in env.targets:
            color = (80, 80, 80) if t['hit'] else (255, 100, 100)
            pygame.draw.circle(screen, color, (int(t['x']), int(t['y'])), int(t['r']), 0 if t['hit'] else 3)
            
        # Draw Ball
        pygame.draw.circle(screen, (255, 255, 255), (int(env.ball['x']), int(env.ball['y'])), env.ball_radius)
        
        # Draw Info
        info_text = font.render(f"Velocity: {velocity:.1f} | Angle: {math.degrees(angle):.1f}°", True, (200, 200, 200))
        screen.blit(info_text, (20, 20))
        
        if done:
            all_hit = all(t['hit'] for t in env.targets)
            msg = "ALL TARGETS HIT!" if all_hit else "Failed"
            color = (255, 255, 100) if all_hit else (255, 50, 50)
            result_text = font.render(msg, True, color)
            screen.blit(result_text, (LOGICAL_WIDTH // 2 - 50, LOGICAL_HEIGHT // 2))
            retry_text = font.render("Press 'R' to Retry", True, (150, 150, 150))
            screen.blit(retry_text, (LOGICAL_WIDTH // 2 - 60, LOGICAL_HEIGHT // 2 + 40))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    replay()
