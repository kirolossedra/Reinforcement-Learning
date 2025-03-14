import pygame
import sys
import math
import os
import random
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BasketBallfellow")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
SKY_BLUE = (135, 206, 235)
WOOD_COLOR = (153, 102, 51)
ORANGE = (255, 165, 0)
NIGHT_SKY = (20, 24, 82)

# Game clock
clock = pygame.time.Clock()
FPS = 60

# Global variable for projectile arc value
projectile_arc = 15  # initial
PROJECTILE_MIN = 5
PROJECTILE_MAX = 30

# Image loader that tries to load from the same directory:
def load_image(name, size=None):
    """Loads an image from the current working directory. 
       If not found, creates a placeholder."""
    try:
        image_path = os.path.join(os.getcwd(), name)
        if not os.path.exists(image_path):
            placeholder = pygame.Surface((64, 64) if size is None else size)
            placeholder.fill(RED)
            pygame.image.save(placeholder, image_path)
            print(f"[WARNING] {name} not found; created a placeholder: {image_path}")
        image = pygame.image.load(image_path).convert_alpha()
        if size:
            image = pygame.transform.scale(image, size)
        return image
    except pygame.error as e:
        print(f"Cannot load image: {name} => {e}")
        placeholder = pygame.Surface((64, 64) if size is None else size)
        placeholder.fill(RED)
        return placeholder

# --- Q-Learning Agent Implementation with Ball Landing Awareness ---
class QLearningAgent:
    def __init__(self, state_bins=10, alpha=0.1, gamma=0.9, epsilon=0.1):
        # We discretize both the player's x and the ball's final x position.
        self.state_bins = state_bins
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration probability

        # Define discrete actions: possible arc values (integers)
        self.actions = list(range(PROJECTILE_MIN, PROJECTILE_MAX + 1))
        # Q-table: keys are state tuples (player_state, ball_state)
        self.Q = {}

    def get_state(self, player_x, ball_final_x):
        # Discretize player's x between left_limit and right_limit.
        left_limit = 50
        right_limit = WIDTH // 2
        norm_px = (player_x - left_limit) / (right_limit - left_limit)
        state_px = int(norm_px * (self.state_bins - 1))
        state_px = max(0, min(state_px, self.state_bins - 1))

        # For the ball's landing x, assume it lies between -50 and WIDTH+50.
        ball_min = -50
        ball_max = WIDTH + 50
        norm_bx = (ball_final_x - ball_min) / (ball_max - ball_min)
        state_bx = int(norm_bx * (self.state_bins - 1))
        state_bx = max(0, min(state_bx, self.state_bins - 1))

        return (state_px, state_bx)

    def ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = [0.0 for _ in self.actions]

    def choose_action(self, state):
        self.ensure_state(state)
        # Epsilon-greedy selection.
        if random.random() < self.epsilon:
            action_index = random.randint(0, len(self.actions) - 1)
        else:
            max_q = max(self.Q[state])
            best_actions = [i for i, q in enumerate(self.Q[state]) if q == max_q]
            action_index = random.choice(best_actions)
        return action_index

    def update(self, state, action_index, reward, next_state):
        self.ensure_state(state)
        self.ensure_state(next_state)
        best_next = max(self.Q[next_state])
        self.Q[state][action_index] += self.alpha * (reward + self.gamma * best_next - self.Q[state][action_index])
        # Debug (optional): print(f"Updated Q{state}[{action_index}] to {self.Q[state][action_index]:.2f}")

# Player class remains the same.
class Player:
    def __init__(self):
        self.image = load_image('player.png', (50, 100))
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.x = WIDTH // 4
        self.y = HEIGHT - self.height - 20  # 20 pixels above ground
        self.speed = 5
        self.jumping = False
        self.jump_velocity = 0
        self.gravity = 0.5
        self.jump_height = 15
        self.left_limit = 50
        self.right_limit = WIDTH // 2
        self.has_ball = True

    def update(self, keys):
        if keys[K_LEFT] and self.x > self.left_limit:
            self.x -= self.speed
        if keys[K_RIGHT] and self.x < self.right_limit - self.width:
            self.x += self.speed
        if keys[K_SPACE] and not self.jumping:
            self.jumping = True
            self.jump_velocity = -self.jump_height
        if self.jumping:
            self.y += self.jump_velocity
            self.jump_velocity += self.gravity
            if self.y >= HEIGHT - self.height - 20:
                self.y = HEIGHT - self.height - 20
                self.jumping = False
                self.jump_velocity = 0

    def draw(self):
        screen.blit(self.image, (self.x, self.y))
        
    def shoot_ball(self):
        global projectile_arc
        if self.has_ball:
            self.has_ball = False
            return Ball(self.x + self.width // 2, self.y, projectile_arc)
        return None

# ScoreBoard class.
class ScoreBoard:
    def __init__(self):
        self.score = 0
        self.font = pygame.font.SysFont('Arial', 32)
    
    def increase(self, val=2):
        self.score += val
    
    def decrease(self, val=1):
        self.score -= val
    
    def draw(self):
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        screen.blit(score_text, (20, 20))

# Ball class now records its final landing position.
class Ball:
    def __init__(self, x, y, arc):
        self.image = load_image('basketball.png', (30, 30))
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.x = x - self.width // 2
        self.y = y
        self.velocity_x = 7
        self.velocity_y = -arc  # use projectile arc value
        self.gravity = 0.5
        self.active = True
        self.scored = False
        self.final_x = self.x + self.width // 2  # default final position
        # For Q-learning updates:
        self.from_rl = False
        self.rl_state = None
        self.rl_action = None
    
    def update(self, basket, scoreboard):
        if not self.active:
            return
        
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += self.gravity

        # Check if ball goes out-of-bounds.
        if self.x > WIDTH + 50:
            self.active = False
            self.final_x = self.x + self.width // 2
            scoreboard.decrease(1)
            return
        elif self.y > HEIGHT + 50 or self.x < -50:
            self.active = False
            self.final_x = self.x + self.width // 2
            return
        
        # Check collision with basket rim area.
        dx = (self.x + self.width/2) - basket.rim_center_x
        dy = (self.y + self.height/2) - basket.rim_center_y
        dist = math.hypot(dx, dy)
        if dist < basket.rim_radius:
            self.scored = True
            # Mark shot as successful and stop the ball.
            self.active = False
            self.final_x = self.x + self.width // 2

    def draw(self):
        if self.active:
            screen.blit(self.image, (self.x, self.y))

# Celebration animation.
class CelebrationAnimation:
    def __init__(self):
        self.duration = 2000  # ms
        self.start_time = pygame.time.get_ticks()
        self.particles = []
        self.num_particles = 20
        for _ in range(self.num_particles):
            x = WIDTH // 2
            y = 50
            angle = random.uniform(0, math.pi)
            speed = random.uniform(2, 5)
            vx = speed * math.cos(angle)
            vy = -abs(speed * math.sin(angle))
            lifetime = random.randint(1000, 2000)
            self.particles.append([x, y, vx, vy, lifetime])
    
    def update(self, dt):
        for particle in self.particles:
            particle[0] += particle[2]
            particle[1] += particle[3]
            particle[3] += 0.1  # gravity
            particle[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]
    
    def draw(self):
        for particle in self.particles:
            pygame.draw.circle(screen, RED, (int(particle[0]), int(particle[1])), 3)
        font = pygame.font.SysFont('Arial', 48, bold=True)
        text = font.render("SCORE!!!", True, RED)
        text_rect = text.get_rect(center=(WIDTH//2, 50))
        screen.blit(text, text_rect)
    
    def is_finished(self):
        return (pygame.time.get_ticks() - self.start_time > self.duration) and len(self.particles) == 0

# Basket class.
class Basket:
    def __init__(self):
        self.x = WIDTH - 140    # backboard left
        self.y = HEIGHT // 2 - 60
        self.backboard_w = 10
        self.backboard_h = 100
        self.rim_radius = 30  
        self.rim_center_x = self.x - self.rim_radius + 5
        self.rim_center_y = self.y + 20
        self.hit_animation_time = 0
    
    def trigger_hit_animation(self):
        self.hit_animation_time = 500  # ms of shaking
    
    def draw(self):
        offset = 0
        if self.hit_animation_time > 0:
            offset = int(5 * math.sin(pygame.time.get_ticks() * 0.05))
            self.hit_animation_time = max(self.hit_animation_time - clock.get_time(), 0)
        backboard_rect = pygame.Rect(self.x + offset, self.y - 20, self.backboard_w, self.backboard_h)
        pygame.draw.rect(screen, WHITE, backboard_rect)
        pygame.draw.rect(screen, BLACK, backboard_rect, 2)
        rim_rect = pygame.Rect(self.rim_center_x - self.rim_radius + offset,
                               self.rim_center_y - self.rim_radius,
                               self.rim_radius * 2,
                               self.rim_radius * 2)
        pygame.draw.arc(screen, RED, rim_rect, 0, math.pi, 4)
        net_offset = 30  
        bottom_radius = int(self.rim_radius * 0.6)
        segments = 6
        top_points = []
        bottom_points = []
        for i in range(segments + 1):
            t = i / segments
            angle = t * math.pi
            tx = (self.rim_center_x + offset) + self.rim_radius * math.cos(angle)
            ty = self.rim_center_y + self.rim_radius * math.sin(angle)
            top_points.append((tx, ty))
            bx = (self.rim_center_x + offset) + bottom_radius * math.cos(angle)
            by = (self.rim_center_y + net_offset) + bottom_radius * math.sin(angle)
            bottom_points.append((bx, by))
        for i in range(segments + 1):
            pygame.draw.line(screen, WHITE, top_points[i], bottom_points[i], 2)
        for i in range(segments):
            pygame.draw.line(screen, WHITE, bottom_points[i], bottom_points[i+1], 2)

# Court class.
class Court:
    def __init__(self):
        self.floor_rect = pygame.Rect(0, HEIGHT - 20, WIDTH, 20)
        self.wall_rect = pygame.Rect(0, 0, WIDTH, HEIGHT - 20)
        self.stars = [(random.randint(0, WIDTH), random.randint(0, (HEIGHT - 20)//2)) for _ in range(50)]
    
    def draw(self, theme):
        if theme == "morning":
            sky_color = SKY_BLUE
        elif theme == "evening":
            sky_color = ORANGE
        elif theme == "night":
            sky_color = NIGHT_SKY
        else:
            sky_color = SKY_BLUE
        pygame.draw.rect(screen, sky_color, self.wall_rect)
        pygame.draw.rect(screen, WOOD_COLOR, self.floor_rect)
        pygame.draw.line(screen, WHITE, (WIDTH // 2, HEIGHT - 20), (WIDTH // 2, HEIGHT - 5), 2)
        pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT - 20), 50, 2)
        if theme == "evening":
            pygame.draw.circle(screen, (255, 255, 0), (WIDTH - 80, 80), 40)
        elif theme == "night":
            for star in self.stars:
                pygame.draw.circle(screen, WHITE, star, 2)

# Main Game loop.
def main():
    global projectile_arc
    player = Player()
    basket = Basket()
    court = Court()
    scoreboard = ScoreBoard()
    balls = []
    celebrations = []
    game_start_time = pygame.time.get_ticks()
    
    # Initialize the Q-learning agent.
    q_agent = QLearningAgent(state_bins=10, alpha=0.1, gamma=0.9, epsilon=0.1)
    rl_shot_timer = 0  # timer in milliseconds
    
    # Initialize last_ball_final_x with a default value (use basket rim center).
    last_ball_final_x = basket.rim_center_x

    running = True
    while running:
        dt = clock.tick(FPS)
        elapsed_time = pygame.time.get_ticks() - game_start_time
        theme_index = (elapsed_time // 60000) % 3
        if theme_index == 0:
            current_theme = "morning"
        elif theme_index == 1:
            current_theme = "evening"
        else:
            current_theme = "night"
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                # Allow human-triggered shot.
                if event.key == K_RETURN and player.has_ball:
                    new_ball = player.shoot_ball()
                    if new_ball:
                        balls.append(new_ball)
                # Allow manual adjustment.
                if event.key == K_w:
                    projectile_arc = min(projectile_arc + 1, PROJECTILE_MAX)
                if event.key == K_s:
                    projectile_arc = max(projectile_arc - 1, PROJECTILE_MIN)
        
        keys = pygame.key.get_pressed()
        player.update(keys)
        
        # --- Q-Learning Agent Shooting: every 2 seconds if the player has the ball.
        rl_shot_timer += dt
        if rl_shot_timer >= 2000:
            if player.has_ball:
                # Current state now includes the last ball landing position.
                state = q_agent.get_state(player.x, last_ball_final_x)
                action_index = q_agent.choose_action(state)
                projectile_arc = q_agent.actions[action_index]
                new_ball = player.shoot_ball()
                if new_ball:
                    new_ball.from_rl = True
                    new_ball.rl_state = state
                    new_ball.rl_action = action_index
                    balls.append(new_ball)
            rl_shot_timer = 0
        
        # Update balls and process outcomes.
        for ball in balls[:]:
            ball.update(basket, scoreboard)
            if not ball.active:
                if ball.from_rl:
                    # Use the ball's final landing position in the next state.
                    next_state = q_agent.get_state(player.x, ball.final_x)
                    reward = 2 if ball.scored else -1
                    q_agent.update(ball.rl_state, ball.rl_action, reward, next_state)
                    # Update last_ball_final_x for the next shot.
                    last_ball_final_x = ball.final_x
                if ball.scored:
                    scoreboard.increase(2)
                    basket.trigger_hit_animation()
                    celebrations.append(CelebrationAnimation())
                balls.remove(ball)
                player.has_ball = True
        
        for celebration in celebrations[:]:
            celebration.update(dt)
            if celebration.is_finished():
                celebrations.remove(celebration)
        
        court.draw(current_theme)
        basket.draw()
        player.draw()
        for ball in balls:
            ball.draw()
        scoreboard.draw()
        
        arc_font = pygame.font.SysFont('Arial', 24)
        arc_text = arc_font.render(f"Arc: {projectile_arc}", True, BLACK)
        arc_rect = arc_text.get_rect(topright=(WIDTH - 20, 20))
        screen.blit(arc_text, arc_rect)
        
        for celebration in celebrations:
            celebration.draw()
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
