import pygame
import numpy as np
import heapq
import random
import time
from collections import deque

TILE_SIZE = 40
GRID_SIZE = 10
WINDOW_SIZE = TILE_SIZE * GRID_SIZE
FPS = 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 50, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DARK_BG = (18, 18, 18) 

POWER_UP_PROB = 0.05
TRAP_PROB = 0.05
SPEED_BOOST = 2
SLOW_DOWN = 0.5

human_score = 0
ai_score = 0


GAME_MODES = ['Classic', 'Time Trial', 'Survival']
selected_mode = None
TIME_LIMIT = 60#for time trial mode
MAZE_SHIFT_INTERVAL = 5  

DIFFICULTY_LEVELS = {1: "Easy", 2: "Medium"}

def generate_maze(grid_size):
    maze = np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.7, 0.3])

    maze[0, 0] = 0
    maze[0, 1] = 0
    maze[1, 0] = 0

    maze[-1, -1] = 0
    maze[-1, -2] = 0
    maze[-2, -1] = 0

    return maze

def add_items_to_maze(maze):
    grid_size = maze.shape[0]
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            if random.random() < POWER_UP_PROB:
                maze[i][j] = 2  
            elif random.random() < TRAP_PROB:
                maze[i][j] = 3  
    return maze

def generate_solvable_maze_with_items(grid_size, verbose=True):
    while True:
        maze = generate_maze(grid_size)
        maze = add_items_to_maze(maze)
        path = a_star(maze, (0, 0), (grid_size - 1, grid_size - 1))
        if path:
            return maze
        else:
            if verbose:
                print("Maze unsolvable, regenerating...")


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    open_set = [(0, start)]
    g_score = {start: 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            tentative_g_score = g_score[current] + 1
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if maze[neighbor[0]][neighbor[1]] == 1:
                    continue
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (tentative_g_score + heuristic(neighbor, goal), neighbor))
    return []

def update_score(player):
    global human_score, ai_score
    if player.color == BLUE:
        human_score += 1
    else:
        ai_score += 1

def minimax(maze, depth, maximizing_player, ai_pos, human_pos, difficulty, recent_positions):
    if depth == 0 or ai_pos == (0, 0) or human_pos == (GRID_SIZE - 1, GRID_SIZE - 1):
        ai_dist = heuristic(ai_pos, (0, 0))
        human_dist = heuristic(human_pos, (GRID_SIZE - 1, GRID_SIZE - 1))
        return human_dist - ai_dist

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    if maximizing_player:
        max_eval = float('-inf')
        for dx, dy in neighbors:
            new_ai_pos = (ai_pos[0] + dx, ai_pos[1] + dy)
            if 0 <= new_ai_pos[0] < GRID_SIZE and 0 <= new_ai_pos[1] < GRID_SIZE:
                if maze[new_ai_pos[0]][new_ai_pos[1]] == 0:
                    dist_bonus = (heuristic(ai_pos, (0, 0)) - heuristic(new_ai_pos, (0, 0))) * 10
                    penalty = -1 if new_ai_pos in recent_positions and new_ai_pos != (0, 0) else 0
                    eval = minimax(maze, depth - 1, False, new_ai_pos, human_pos, difficulty, recent_positions) + dist_bonus + penalty
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for dx, dy in neighbors:
            new_human_pos = (human_pos[0] + dx, human_pos[1] + dy)
            if 0 <= new_human_pos[0] < GRID_SIZE and 0 <= new_human_pos[1] < GRID_SIZE:
                if maze[new_human_pos[0]][new_human_pos[1]] == 0:
                    eval = minimax(maze, depth - 1, True, ai_pos, new_human_pos, difficulty, recent_positions)
                    min_eval = min(min_eval, eval)
        return min_eval

def get_ai_candidate_moves(maze, ai_pos, goal):
    path = a_star(maze, ai_pos, goal)#finding shortest path from current pos
    if not path:
        return []
    candidates = []
    if path:
        next_step = path[0]
        candidates.append(next_step)

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in neighbors:
        npos = (ai_pos[0] + dx, ai_pos[1] + dy)
        if 0 <= npos[0] < GRID_SIZE and 0 <= npos[1] < GRID_SIZE:
            if maze[npos[0]][npos[1]] == 0 and npos not in candidates:
                candidates.append(npos)
    return candidates

class Player:
    def __init__(self, x, y, color, is_ai=False, difficulty=1):
        self.x = x
        self.y = y
        self.color = color
        self.is_ai = is_ai
        self.speed = 1
        self.difficulty = difficulty
        self.recent_positions = deque(maxlen=6)
        self.last_move_time = time.time()

    def move(self, maze, human_pos=None):
        ai_move_interval = 0.2 #secs(adjust ai speed)
    
        if self.is_ai and (time.time() - self.last_move_time < ai_move_interval):
            return
    
        self.last_move_time = time.time()
    
        if self.is_ai:
            candidates = get_ai_candidate_moves(maze, (self.x, self.y), (0, 0))
        if not candidates:
            return  

        for candidate in candidates:
            if candidate == (0, 0):
                self.recent_positions.append((self.x, self.y))
                self.x, self.y = candidate
                return

        best_score = float('-inf')
        best_move = (self.x, self.y)
        depth = self.difficulty + 1

        for candidate in candidates:
            penalty = -1 if candidate in self.recent_positions and candidate != (0, 0) else 0
            score = minimax(maze, depth, False, candidate, human_pos, self.difficulty, self.recent_positions) + penalty
            if score > best_score:
                best_score = score
                best_move = candidate

        self.recent_positions.append((self.x, self.y))
        self.x, self.y = best_move

        if maze[self.x][self.y] == 2:
            self.speed = SPEED_BOOST
            update_score(self)
            maze[self.x][self.y] = 0
        elif maze[self.x][self.y] == 3:
            self.speed = SLOW_DOWN
            maze[self.x][self.y] = 0


    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.y * TILE_SIZE, self.x * TILE_SIZE, TILE_SIZE, TILE_SIZE))


def show_countdown():
    countdown_font = pygame.font.SysFont(None, 100)
    for count in ["3", "2", "1", "Go!"]:
        win.fill(DARK_BG)  
        text = countdown_font.render(count, True, GREEN)
        rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        win.blit(text, rect)
        pygame.display.update()
        pygame.time.wait(300)
pygame.init()
win = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("The Moving Maze")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

def select_mode():
    global selected_mode
    selecting = True
    while selecting:
        win.fill(DARK_BG)
        texts = [
            "Select Mode:",
            "1 - Classic",
            "2 - Time Trial",
            "3 - Survival"
        ]
        for i, text in enumerate(texts):
            txt_surf = font.render(text, True, WHITE)
            win.blit(txt_surf, (50, 50 + i * 60))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    selected_mode = 'Classic'
                    selecting = False
                elif event.key == pygame.K_2:
                    selected_mode = 'Time Trial'
                    selecting = False
                elif event.key == pygame.K_3:
                    selected_mode = 'Survival'
                    selecting = False

select_mode()

def select_difficulty():
    selecting = True
    difficulty = 1
    while selecting:
        win.fill(DARK_BG)
        texts = [
            "Select Difficulty:",
            "1 - Easy",
            "2 - Medium"
        ]
        for i, text in enumerate(texts):
            txt_surf = font.render(text, True, WHITE)
            win.blit(txt_surf, (50, 50 + i * 60))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    difficulty = 1
                    selecting = False
                elif event.key == pygame.K_2:
                    difficulty = 2
                    selecting = False
    return difficulty

ai_difficulty = select_difficulty()

maze = generate_solvable_maze_with_items(GRID_SIZE, verbose=False)
human = Player(0, 0, BLUE)
ai = Player(GRID_SIZE - 1, GRID_SIZE - 1, RED, is_ai=True, difficulty=ai_difficulty)

show_countdown()

running = True
turns = 0
start_time = time.time()

while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if selected_mode == 'Time Trial':
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            print("Time's up! You lost!")
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and human.x > 0 and maze[human.x - 1][human.y] == 0:
        human.x -= human.speed
        turns += 1
    if keys[pygame.K_DOWN] and human.x < GRID_SIZE - 1 and maze[human.x + 1][human.y] == 0:
        human.x += human.speed
        turns += 1
    if keys[pygame.K_LEFT] and human.y > 0 and maze[human.x][human.y - 1] == 0:
        human.y -= human.speed
        turns += 1
    if keys[pygame.K_RIGHT] and human.y < GRID_SIZE - 1 and maze[human.x][human.y + 1] == 0:
        human.y += human.speed
        turns += 1

    #logic defined for shifting of the maze
    if turns > 0 and turns % MAZE_SHIFT_INTERVAL == 0:
        maze = generate_solvable_maze_with_items(GRID_SIZE)
        human.path = []
        ai.path = []

    ai.move(maze, (human.x, human.y))

    if (human.x, human.y) == (GRID_SIZE - 1, GRID_SIZE - 1):
        print("Human wins!")
        running = False
    if (ai.x, ai.y) == (0, 0):
        print("AI wins!")
        running = False

    #drawing the maze
    win.fill(DARK_BG)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color = BLACK if maze[i][j] == 1 else DARK_BG
            if maze[i][j] == 2:
                color = GREEN
            elif maze[i][j] == 3:
                color = RED
            pygame.draw.rect(win, color, (j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE))

   
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(win, (50, 50, 50), (0, i * TILE_SIZE), (WINDOW_SIZE, i * TILE_SIZE))
        pygame.draw.line(win, (50, 50, 50), (i * TILE_SIZE, 0), (i * TILE_SIZE, WINDOW_SIZE))

    human.draw(win)
    ai.draw(win)

    if selected_mode == 'Time Trial':
        remaining = max(0, TIME_LIMIT - (time.time() - start_time))
        pygame.display.set_caption(f"Mode: {selected_mode} | Difficulty: {DIFFICULTY_LEVELS[ai.difficulty]} | Time Left: {int(remaining)}s")
    else:
        pygame.display.set_caption(f"Human: {human_score} | AI: {ai_score} | Mode: {selected_mode} | Difficulty: {DIFFICULTY_LEVELS[ai.difficulty]}")

    pygame.display.update()

pygame.quit()
