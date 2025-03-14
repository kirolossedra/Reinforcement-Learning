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

# --- Helper: Estimate the predicted drop location of the ball ---
def estimate_drop_location(ball):
    """
    Estimate where the ball will hit ground (level = HEIGHT-20) using projectile motion.
    Uses the quadratic equation to solve for time until y reaches ground.
    """
    ground_y = HEIGHT - 20
    # If ball is already at or below ground, return its current center x.
    if ball.y >= ground_y:
        return ball.x + ball.width/2
    a = 0.5 * ball.gravity
    b = ball.velocity_y
    c = ball.y - ground_y
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return ball.x + ball.width/2
    t = (-b + math.sqrt(discriminant)) / (2 * a)  # positive root
    predicted_x = ball.x + ball.width/2 + ball.velocity_x * t
    return predicted_x

class QLearningAgent:
    def __init__(self, state_bins=20, alpha=0.1, gamma=0.95, epsilon=0.3, epsilon_decay=0.9995, min_epsilon=0.05):
        # Increased state bins for finer discretization
        self.state_bins = state_bins
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor (increased for longer-term planning)
        self.epsilon = epsilon  # exploration probability
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Actions: possible arc values
        self.actions = list(range(PROJECTILE_MIN, PROJECTILE_MAX + 1))
        
        # Q-table with enhanced state representation
        self.Q = {}
        
        # Experience replay buffer to improve learning stability
        self.replay_buffer = []
        self.replay_buffer_size = 1000
        self.batch_size = 32
        
        # Track shots and success rates for different conditions
        self.shot_history = {}  # {state: [(action, reward), ...]}
        
        # Estimated basket position (to be learned)
        self.estimated_basket_x = WIDTH - 140  # Initial estimate based on game default
        self.basket_position_samples = []
        
        # Initialize with some reasonable values
        self._init_q_values()
    
    def _init_q_values(self):
        # Pre-initialize Q-values with some reasonable estimates
        for player_x in range(self.state_bins):
            for drop_x in range(self.state_bins):
                state = (player_x, drop_x)
                if state not in self.Q:
                    # Initialize with slight preference for middle arc values
                    self.Q[state] = [0.0] * len(self.actions)
                    for i, action in enumerate(self.actions):
                        # Slight bias toward middle arc values
                        self.Q[state][i] = -abs(action - (PROJECTILE_MIN + PROJECTILE_MAX) / 2) / 10.0
    
    def update_basket_estimate(self, scored, drop_x):
        """
        Update the estimated basket position based on ball outcomes
        """
        # If the ball scored, this is a very accurate indication of basket position
        if scored:
            self.basket_position_samples.append(drop_x)
            # Keep recent samples only
            if len(self.basket_position_samples) > 5:
                self.basket_position_samples.pop(0)
            # Update estimate based on recent samples
            if self.basket_position_samples:
                self.estimated_basket_x = sum(self.basket_position_samples) / len(self.basket_position_samples)
    
    def get_state(self, player_x, drop_x):
        # Discretize player's x between left_limit and right_limit
        left_limit = 50
        right_limit = WIDTH // 2
        norm_px = max(0, min(1, (player_x - left_limit) / (right_limit - left_limit)))
        state_px = int(norm_px * (self.state_bins - 1))
        
        # Discretize drop_x relative to estimated basket position
        # This makes learning transfer across absolute positions
        relative_drop = drop_x - self.estimated_basket_x
        # Scale to reasonable range (-300 to +300 pixels from basket)
        norm_drop = max(0, min(1, (relative_drop + 300) / 600))
        state_drop = int(norm_drop * (self.state_bins - 1))
        
        return (state_px, state_drop)
    
    def ensure_state(self, state):
        if state not in self.Q:
            # Initialize with values that reflect what we've learned about similar states
            self.Q[state] = [0.0] * len(self.actions)
            
            # Try to find similar states to initialize from
            similar_states = []
            for s in self.Q:
                # Calculate similarity (lower is more similar)
                diff = sum(abs(a - b) for a, b in zip(s, state))
                if diff <= 2:  # States that are "close enough"
                    similar_states.append((diff, s))
            
            # If we found similar states, initialize from them
            if similar_states:
                similar_states.sort()  # Sort by similarity
                best_similar = similar_states[0][1]
                for i in range(len(self.actions)):
                    self.Q[state][i] = self.Q[best_similar][i]
            else:
                # Initialize with slight preference for middle arc values
                for i, action in enumerate(self.actions):
                    # Slight bias toward middle arc values
                    self.Q[state][i] = -abs(action - (PROJECTILE_MIN + PROJECTILE_MAX) / 2) / 10.0
    
    def choose_action(self, state):
        self.ensure_state(state)
        
        # If we have shot history for this state, analyze success rates
        if state in self.shot_history and random.random() > self.epsilon / 2:
            # Use history to make an informed choice occasionally
            action_rewards = {}
            for action_idx, reward in self.shot_history[state]:
                if action_idx not in action_rewards:
                    action_rewards[action_idx] = []
                action_rewards[action_idx].append(reward)
            
            # Find action with best average reward
            best_action = None
            best_avg = float('-inf')
            for action_idx, rewards in action_rewards.items():
                avg = sum(rewards) / len(rewards)
                if avg > best_avg:
                    best_avg = avg
                    best_action = action_idx
            
            if best_action is not None and best_avg > 0:
                return best_action
        
        # Standard ε-greedy policy
        if random.random() < self.epsilon:
            # Sometimes select actions near current best instead of completely random
            if random.random() < 0.7 and sum(self.Q[state]) != 0:
                # Find best action index
                best_idx = self.Q[state].index(max(self.Q[state]))
                # Choose a nearby action (±2)
                low = max(0, best_idx - 2)
                high = min(len(self.actions) - 1, best_idx + 2)
                action_index = random.randint(low, high)
            else:
                action_index = random.randint(0, len(self.actions) - 1)
        else:
            max_q = max(self.Q[state])
            # In case of ties, choose randomly among best
            best_actions = [i for i, q in enumerate(self.Q[state]) if q == max_q]
            action_index = random.choice(best_actions)
        
        return action_index
    
    def add_to_replay(self, state, action_index, reward, next_state):
        # Add experience to replay buffer
        self.replay_buffer.append((state, action_index, reward, next_state))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)
        
        # Track shot history for this state
        if state not in self.shot_history:
            self.shot_history[state] = []
        self.shot_history[state].append((action_index, reward))
        # Keep shot history manageable
        if len(self.shot_history[state]) > 10:
            self.shot_history[state].pop(0)
    
    def replay_experiences(self):
        # Experience replay - learn from multiple past experiences
        if len(self.replay_buffer) < self.batch_size or len(self.replay_buffer) < 10:
            return
        
        # Use smaller batch if we don't have enough data yet
        actual_batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer, actual_batch_size)
        for state, action_index, reward, next_state in batch:
            self.update_q_value(state, action_index, reward, next_state)
    
    def update_q_value(self, state, action_index, reward, next_state):
        self.ensure_state(state)
        self.ensure_state(next_state)
        
        # Standard Q-learning update
        best_next = max(self.Q[next_state])
        self.Q[state][action_index] += self.alpha * (
            reward + self.gamma * best_next - self.Q[state][action_index]
        )
    
    def update(self, state, action_index, reward, next_state, scored=False, predicted_drop=None):
        # Update basket position estimate if we have drop information
        if predicted_drop is not None:
            self.update_basket_estimate(scored, predicted_drop)
        
        # Add to replay buffer
        self.add_to_replay(state, action_index, reward, next_state)
        
        # Direct update for this experience
        self.update_q_value(state, action_index, reward, next_state)
        
        # Learn from past experiences
        self.replay_experiences()
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def adjust_arc_for_distance(self, state, predicted_drop):
        """
        Intelligently suggest arc adjustments based on physical principles
        and current observed drop location vs. estimated basket position.
        """
        # Get current best arc from Q-values
        self.ensure_state(state)
        best_action_index = self.Q[state].index(max(self.Q[state]))
        current_arc = self.actions[best_action_index]
        
        # If we're undershooting consistently
        if predicted_drop < self.estimated_basket_x - 30:
            # Increase arc to extend distance
            return min(current_arc + 1, PROJECTILE_MAX)
        
        # If we're overshooting consistently
        elif predicted_drop > self.estimated_basket_x + 30:
            # Decrease arc to reduce distance
            return max(current_arc - 1, PROJECTILE_MIN)
        
        # If we're close, stick with current best
        return current_arc
# --- Player Class ---
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

# --- ScoreBoard Class ---
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

# --- Ball Class with Estimation for Drop ---
class Ball:
    def __init__(self, x, y, arc):
        self.image = load_image('basketball.png', (30, 30))
        self.width = self.image.get_width()
        self.height = self.image.get_height()
        self.x = x - self.width // 2
        self.y = y
        self.velocity_x = 7
        self.velocity_y = -arc  # projectile arc used here
        self.gravity = 0.5
        self.active = True
        self.scored = False
        # Store the predicted drop location (will be updated)
        self.predicted_drop = self.x + self.width/2  
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

        # Update the predicted drop location on the fly.
        self.predicted_drop = estimate_drop_location(self)

        # Check if ball goes out-of-bounds.
        if self.x > WIDTH + 50:
            self.active = False
            scoreboard.decrease(1)
            return
        elif self.y > HEIGHT + 50 or self.x < -50:
            self.active = False
            return
        
        # Check collision with basket rim area.
        dx = (self.x + self.width/2) - basket.rim_center_x
        dy = (self.y + self.height/2) - basket.rim_center_y
        dist = math.hypot(dx, dy)
        if dist < basket.rim_radius:
            self.scored = True
            self.active = False

    def draw(self):
        if self.active:
            screen.blit(self.image, (self.x, self.y))

# --- Celebration Animation ---
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

# --- Basket Class ---
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

# --- Court Class ---
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

# --- Main Game Loop ---
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
    q_agent = QLearningAgent(state_bins=10, alpha=0.1, gamma=0.9, epsilon=0.3)
    rl_shot_timer = 0  # milliseconds timer

    # Initialize a default estimated drop location.
    last_predicted_drop = basket.rim_center_x

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
        
        # --- Q-Learning Agent Shooting (every 2 seconds) ---
        rl_shot_timer += dt
        if rl_shot_timer >= 2000:
            if player.has_ball:
                # The current state includes the player's x and the last predicted drop location.
                state = q_agent.get_state(player.x, last_predicted_drop)
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
                    # Compute a fresh estimated drop location using physics.
                    predicted_drop = estimate_drop_location(ball)
                    # Next state: current player position and the estimated drop.
                    next_state = q_agent.get_state(player.x, predicted_drop)
                    # Reward: +2 if scored; if missed, penalize proportionally to the error from basket center.
                    if ball.scored:
                        reward = 2
                    else:
                        error = abs(predicted_drop - basket.rim_center_x)
                        reward = - (error / 100.0)
                    q_agent.update(ball.rl_state, ball.rl_action, reward, next_state)
                    # Update the last predicted drop for the next shot.
                    last_predicted_drop = predicted_drop
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
