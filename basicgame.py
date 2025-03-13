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
            # Create a placeholder image if file doesn't exist
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
        # Return a placeholder in case of error
        placeholder = pygame.Surface((64, 64) if size is None else size)
        placeholder.fill(RED)
        return placeholder

# Player class
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
        # Horizontal movement
        if keys[K_LEFT] and self.x > self.left_limit:
            self.x -= self.speed
        if keys[K_RIGHT] and self.x < self.right_limit - self.width:
            self.x += self.speed
        
        # Jumping
        if keys[K_SPACE] and not self.jumping:
            self.jumping = True
            self.jump_velocity = -self.jump_height
        
        if self.jumping:
            self.y += self.jump_velocity
            self.jump_velocity += self.gravity
            
            # Check if landed
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

# ScoreBoard class
class ScoreBoard:
    def __init__(self):
        self.score = 0
        self.font = pygame.font.SysFont('Arial', 32)
    
    def increase(self, val=2):
        self.score += val  # default: 2 points for a basket
    
    def decrease(self, val=1):
        self.score -= val
    
    def draw(self):
        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        screen.blit(score_text, (20, 20))

# Ball class with variable arc
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
    
    def update(self, basket, scoreboard):
        if not self.active:
            return
            
        # Update position (projectile motion)
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.velocity_y += self.gravity
        
        # Check if ball went out of bounds
        # Deduct score only if ball goes beyond the right boundary.
        if self.x > WIDTH + 50:
            self.active = False
            scoreboard.decrease(1)
            return
        elif self.y > HEIGHT + 50 or self.x < -50:
            self.active = False
            return
        
        # Check collision with basket rim area
        dx = (self.x + self.width/2) - basket.rim_center_x
        dy = (self.y + self.height/2) - basket.rim_center_y
        dist = math.hypot(dx, dy)
        if dist < basket.rim_radius:  # scored if inside rim radius
            self.scored = True

    def draw(self):
        if self.active:
            screen.blit(self.image, (self.x, self.y))

# Celebration animation for scoring
class CelebrationAnimation:
    def __init__(self):
        self.duration = 2000  # Duration in milliseconds
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
            particle[3] += 0.1  # simulate gravity
            particle[4] -= dt
        self.particles = [p for p in self.particles if p[4] > 0]
    
    def draw(self):
        # Particles
        for particle in self.particles:
            pygame.draw.circle(screen, RED, (int(particle[0]), int(particle[1])), 3)
        # "SCORE!!!" text
        font = pygame.font.SysFont('Arial', 48, bold=True)
        text = font.render("SCORE!!!", True, RED)
        text_rect = text.get_rect(center=(WIDTH//2, 50))
        screen.blit(text, text_rect)
    
    def is_finished(self):
        return (pygame.time.get_ticks() - self.start_time > self.duration) and len(self.particles) == 0

# Basket class with improved rim & net drawing
class Basket:
    def __init__(self):
        # Position & size
        self.x = WIDTH - 140    # backboard left
        self.y = HEIGHT // 2 - 60
        self.backboard_w = 10
        self.backboard_h = 100
        
        # Rim settings
        self.rim_radius = 30  # radius of the half-circle rim
        # We'll place the rim center a bit away from the backboard
        self.rim_center_x = self.x - self.rim_radius + 5
        self.rim_center_y = self.y + 20
        self.hit_animation_time = 0  # in ms
    
    def trigger_hit_animation(self):
        self.hit_animation_time = 500  # 500 ms of shaking
    
    def draw(self):
        offset = 0
        if self.hit_animation_time > 0:
            # Shake effect
            offset = int(5 * math.sin(pygame.time.get_ticks() * 0.05))
            self.hit_animation_time = max(self.hit_animation_time - clock.get_time(), 0)
        
        # Draw backboard
        backboard_rect = pygame.Rect(
            self.x + offset, 
            self.y - 20, 
            self.backboard_w, 
            self.backboard_h
        )
        pygame.draw.rect(screen, WHITE, backboard_rect)
        pygame.draw.rect(screen, BLACK, backboard_rect, 2)
        
        # Draw rim (half circle) => arc from 0 to pi
        rim_rect = pygame.Rect(
            self.rim_center_x - self.rim_radius + offset,
            self.rim_center_y - self.rim_radius,
            self.rim_radius * 2,
            self.rim_radius * 2
        )
        pygame.draw.arc(screen, RED, rim_rect, 0, math.pi, 4)
        
        # Draw net below the rim as a second arc + connecting lines
        net_offset = 30  # how far below the top arc
        bottom_radius = int(self.rim_radius * 0.6)
        
        # Top arc points (the rim) and bottom arc points
        segments = 6
        top_points = []
        bottom_points = []
        for i in range(segments + 1):
            t = i / segments
            angle = t * math.pi  # from 0 to pi
            # top arc
            tx = (self.rim_center_x + offset) + self.rim_radius * math.cos(angle)
            ty = self.rim_center_y + self.rim_radius * math.sin(angle)
            top_points.append((tx, ty))
            # bottom arc
            bx = (self.rim_center_x + offset) + bottom_radius * math.cos(angle)
            by = (self.rim_center_y + net_offset) + bottom_radius * math.sin(angle)
            bottom_points.append((bx, by))
        
        # Draw net lines from top arc to bottom arc
        for i in range(segments + 1):
            pygame.draw.line(screen, WHITE, top_points[i], bottom_points[i], 2)
        
        # Draw the bottom arc
        for i in range(segments):
            pygame.draw.line(screen, WHITE, bottom_points[i], bottom_points[i+1], 2)

# Court class with variable scenes (morning, evening, night)
class Court:
    def __init__(self):
        self.floor_rect = pygame.Rect(0, HEIGHT - 20, WIDTH, 20)
        self.wall_rect = pygame.Rect(0, 0, WIDTH, HEIGHT - 20)
        # Pre-generate stars for the night theme
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
        
        # Draw sky
        pygame.draw.rect(screen, sky_color, self.wall_rect)
        # Draw floor (court)
        pygame.draw.rect(screen, WOOD_COLOR, self.floor_rect)
        
        # Draw center court lines
        pygame.draw.line(screen, WHITE, (WIDTH // 2, HEIGHT - 20), (WIDTH // 2, HEIGHT - 5), 2)
        pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT - 20), 50, 2)
        
        # Additional elements per theme
        if theme == "evening":
            # Draw a setting sun near the top-right corner
            pygame.draw.circle(screen, (255, 255, 0), (WIDTH - 80, 80), 40)
        elif theme == "night":
            # Draw stars
            for star in self.stars:
                pygame.draw.circle(screen, WHITE, star, 2)

# Main Game loop
def main():
    global projectile_arc
    # Create game objects
    player = Player()
    basket = Basket()
    court = Court()
    scoreboard = ScoreBoard()
    balls = []
    celebrations = []
    game_start_time = pygame.time.get_ticks()
    
    running = True
    while running:
        dt = clock.tick(FPS)
        
        # Cycle through themes every 1 minute
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
                if event.key == K_RETURN and player.has_ball:
                    new_ball = player.shoot_ball()
                    if new_ball:
                        balls.append(new_ball)
                # Adjust projectile arc value in real time:
                if event.key == K_w:
                    projectile_arc = min(projectile_arc + 1, PROJECTILE_MAX)
                if event.key == K_s:
                    projectile_arc = max(projectile_arc - 1, PROJECTILE_MIN)
        
        keys = pygame.key.get_pressed()
        player.update(keys)
        
        # Update balls and check for scoring
        for ball in balls[:]:
            ball.update(basket, scoreboard)
            if not ball.active:
                # If ball is inactive, it either went out or scored
                if ball.scored:
                    scoreboard.increase(2)
                    basket.trigger_hit_animation()
                    celebrations.append(CelebrationAnimation())
                balls.remove(ball)
                player.has_ball = True
        
        # Update celebration animations
        for celebration in celebrations[:]:
            celebration.update(dt)
            if celebration.is_finished():
                celebrations.remove(celebration)
        
        # Draw everything
        court.draw(current_theme)
        basket.draw()
        player.draw()
        for ball in balls:
            ball.draw()
        scoreboard.draw()
        
        # Draw the projectile arc value at top-right
        arc_font = pygame.font.SysFont('Arial', 24)
        arc_text = arc_font.render(f"Arc: {projectile_arc}", True, BLACK)
        arc_rect = arc_text.get_rect(topright=(WIDTH - 20, 20))
        screen.blit(arc_text, arc_rect)
        
        # Celebration animations on top
        for celebration in celebrations:
            celebration.draw()
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
