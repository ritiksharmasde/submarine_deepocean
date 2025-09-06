import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random, heapq

GRID_SIZE = 12         # 12x12x12 grid
STATIC_OBSTACLES = 80  # number of random static obstacles
DYNAMIC_COUNT = 6      # number of dynamic obstacles (moving)
FRAMES = 300
INTERVAL_MS = 200

# ---------------- Maps ----------------
# SLAM map: probabilities [0..1] where 0.5 = unknown
slam_map = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), 0.5)

# Real ocean: 0 free, 1 obstacle
ocean = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=int)

# Place static obstacles randomly
placed = 0
while placed < STATIC_OBSTACLES:
    x, y, z = (random.randrange(GRID_SIZE) for _ in range(3))
    # avoid placing on center start or near goal
    if (x, y, z) != (GRID_SIZE//2, GRID_SIZE//2, GRID_SIZE//2) and ocean[x,y,z] == 0:
        ocean[x, y, z] = 1
        placed += 1

# Dynamic obstacles: random positions that are free
dynamic_obstacles = []
while len(dynamic_obstacles) < DYNAMIC_COUNT:
    x, y, z = (random.randrange(GRID_SIZE) for _ in range(3))
    if ocean[x,y,z] == 0 and (x,y,z) != (GRID_SIZE//2, GRID_SIZE//2, GRID_SIZE//2):
        dynamic_obstacles.append((x,y,z))
        ocean[x,y,z] = 1

# Sub and goal
sub_pos = [GRID_SIZE//2, GRID_SIZE//2, GRID_SIZE//2]
goal_pos = [GRID_SIZE-2, GRID_SIZE-2, GRID_SIZE-2]

path = [tuple(sub_pos)]
planned_path = []

# 6-connected moves: (dx,dy,dz)
directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
dir_names = ["+X","-X","+Y","-Y","+Z","-Z"]

# ---------------- A* (3D) ----------------
def astar_3d(grid, start, goal):
    rows, cols, deps = grid.shape
    def heuristic(a,b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])  # Manhattan in 3D

    open_set = []
    heapq.heappush(open_set, (heuristic(start,goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            p = [current]
            while parent:
                p.append(parent)
                parent = came_from.get(parent)
            return p[::-1]
        came_from[current] = parent
        for dx,dy,dz in directions:
            nx,ny,nz = current[0]+dx, current[1]+dy, current[2]+dz
            if 0 <= nx < rows and 0 <= ny < cols and 0 <= nz < deps and grid[nx,ny,nz] == 0:
                tg = g + 1
                if tg < g_score.get((nx,ny,nz), float("inf")):
                    g_score[(nx,ny,nz)] = tg
                    f_score = tg + heuristic((nx,ny,nz), goal)
                    heapq.heappush(open_set, (f_score, tg, (nx,ny,nz), current))
    return []

# ---------------- Sonar (6 directions) ----------------
def sonar_readings_3d(pos, max_range=None):
    x,y,z = pos
    readings = {}
    # For each direction, count free cells until obstacle/boundary (or max_range if given)
    for name, (dx,dy,dz) in zip(dir_names, directions):
        d = 0
        nx,ny,nz = x+dx, y+dy, z+dz
        while 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
            if ocean[nx,ny,nz] == 1:
                break
            d += 1
            if max_range is not None and d >= max_range:
                break
            nx += dx; ny += dy; nz += dz
        readings[name] = d
    return readings

# ---------------- Dynamic obstacles move in 3D ----------------
def move_obstacles_3d():
    global dynamic_obstacles, ocean
    # clear old positions
    for (x,y,z) in dynamic_obstacles:
        # only clear if it's still marked as dynamic (avoid clearing a static obstacle accidentally)
        if ocean[x,y,z] == 1:
            ocean[x,y,z] = 0
    new_positions = []
    for (x,y,z) in dynamic_obstacles:
        # pick a move including possibility of staying put
        moves = directions + [(0,0,0)]
        dx,dy,dz = random.choice(moves)
        nx,ny,nz = x+dx, y+dy, z+dz
        # keep inside grid and avoid submarine & goal
        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE
            and (nx,ny,nz) != tuple(sub_pos) and (nx,ny,nz) != tuple(goal_pos)
            and ocean[nx,ny,nz] == 0):  # avoid stepping onto static/dynamic obstacles
            new_positions.append((nx,ny,nz))
        else:
            new_positions.append((x,y,z))
    # mark new positions
    for (x,y,z) in new_positions:
        ocean[x,y,z] = 1
    dynamic_obstacles = new_positions

# ---------------- Visualization helpers ----------------
def get_occupied_points(grid):
    xs, ys, zs = np.where(grid == 1)
    return list(zip(xs, ys, zs))

def slam_points_from_map(smap):
    occ = np.argwhere(smap >= 0.9)  # predicted obstacles
    free = np.argwhere(smap <= 0.1) # predicted free
    unknown = np.argwhere((smap > 0.1) & (smap < 0.9))
    return occ, free, unknown

# ---------------- Update (animation) ----------------
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

def update(frame):
    global sub_pos, slam_map, planned_path

    # 1) Move dynamic obstacles first
    move_obstacles_3d()

    # 2) Sonar and SLAM update
    x,y,z = sub_pos
    readings = sonar_readings_3d(sub_pos, max_range=None)  # unlimited range in this version
    for name, dist in readings.items():
        dx,dy,dz = directions[dir_names.index(name)]
        # mark free cells along beam
        for i in range(1, dist+1):
            nx,ny,nz = x+dx*i, y+dy*i, z+dz*i
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                # conservative update: set to free (0)
                slam_map[nx,ny,nz] = 0.0
        # mark obstacle cell if within grid
        nx,ny,nz = x+dx*(dist+1), y+dy*(dist+1), z+dz*(dist+1)
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
            slam_map[nx,ny,nz] = 1.0

    # 3) Replan if needed
    if not planned_path or planned_path[0] != tuple(sub_pos):
        planned_path = astar_3d(ocean, tuple(sub_pos), tuple(goal_pos))

    # 4) Follow path (one step)
    if len(planned_path) > 1:
        planned_path.pop(0)
        next_step = planned_path[0]
        sub_pos[0], sub_pos[1], sub_pos[2] = next_step
        path.append(tuple(sub_pos))

    # 5) Plot real world (left)
    ax1.cla()
    ax1.set_title("Real Ocean (3D)")
    ax1.set_xlim(0, GRID_SIZE); ax1.set_ylim(0, GRID_SIZE); ax1.set_zlim(0, GRID_SIZE)
    occ = get_occupied_points(ocean)
    if occ:
        xs, ys, zs = zip(*occ)
        ax1.scatter(xs, ys, zs, marker='s', s=40, alpha=0.8)  # obstacles
    # dynamic obstacles highlight
    dxs, dys, dzs = zip(*dynamic_obstacles)
    ax1.scatter(dxs, dys, dzs, color='red', s=80, label='dynamic')
    # path
    if path:
        px,py,pz = zip(*path)
        ax1.plot(px, py, pz, color='orange')
    # sub and goal
    ax1.scatter(sub_pos[0], sub_pos[1], sub_pos[2], color='yellow', s=150, label='sub')
    ax1.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color='green', s=120, marker='X', label='goal')
    ax1.legend(loc='upper left', fontsize='small')

    # 6) Plot SLAM estimate (right)
    ax2.cla()
    ax2.set_title("SLAM Map (3D Estimate)")
    ax2.set_xlim(0, GRID_SIZE); ax2.set_ylim(0, GRID_SIZE); ax2.set_zlim(0, GRID_SIZE)
    occ_s, free_s, unknown_s = slam_points_from_map(slam_map)
    if occ_s.size:
        ax2.scatter(occ_s[:,0], occ_s[:,1], occ_s[:,2], c='red', marker='s', s=40, alpha=0.9, label='predicted obstacle')
    if free_s.size:
        ax2.scatter(free_s[:,0], free_s[:,1], free_s[:,2], c='blue', marker='o', s=20, alpha=0.6, label='predicted free')
    if unknown_s.size:
        ax2.scatter(unknown_s[:,0], unknown_s[:,1], unknown_s[:,2], c='gray', marker='.', s=8, alpha=0.3, label='unknown')
    # sub & goal
    ax2.scatter(sub_pos[0], sub_pos[1], sub_pos[2], color='yellow', s=150)
    ax2.scatter(goal_pos[0], goal_pos[1], goal_pos[2], color='green', s=120, marker='X')
    ax2.legend(loc='upper left', fontsize='small')

# ---------------- Run animation ----------------
ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL_MS, repeat=False)
plt.show()
