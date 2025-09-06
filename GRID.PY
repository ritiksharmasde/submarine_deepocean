import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random, heapq

# Grid size
GRID_SIZE = 20
slam_map = np.full((GRID_SIZE, GRID_SIZE), 0.5)  # 0.5 = unknown

# Create ocean grid (0 = free water, 1 = obstacle)
ocean = np.zeros((GRID_SIZE, GRID_SIZE))
for _ in range(30):  # static obstacles
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    ocean[x, y] = 1

# Dynamic obstacles
dynamic_obstacles = [(5, 5), (10, 15), (15, 5)]
for x, y in dynamic_obstacles:
    ocean[x, y] = 1

# Submarine start + goal
sub_pos = [GRID_SIZE//2, GRID_SIZE//2]
goal_pos = [GRID_SIZE-2, GRID_SIZE-2]

# Store path
path = [tuple(sub_pos)]
planned_path = []  # path computed with A*

# Moves
moves = {"UP": (0, 1), "DOWN": (0, -1), "RIGHT": (1, 0), "LEFT": (-1, 0)}

# ---------- A* ----------
def astar(grid, start, goal):
    rows, cols = grid.shape
    directions = [(0,1),(0,-1),(1,0),(-1,0)]

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set=[]
    heapq.heappush(open_set,(heuristic(start,goal),0,start,None))
    came_from,g_score={}, {start:0}

    while open_set:
        f,g,current,parent=heapq.heappop(open_set)
        if current==goal:
            p=[current]
            while parent:
                p.append(parent)
                parent=came_from.get(parent)
            return p[::-1]

        came_from[current]=parent
        for dx,dy in directions:
            nx,ny=current[0]+dx,current[1]+dy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==0:
                tg=g+1
                if tg<g_score.get((nx,ny),float("inf")):
                    g_score[(nx,ny)]=tg
                    f_score=tg+heuristic((nx,ny),goal)
                    heapq.heappush(open_set,(f_score,tg,(nx,ny),current))
    return []

# ---------- Sonar ----------
def sonar_readings(pos):
    x,y=pos
    readings={}
    # UP
    d=0
    for ny in range(y+1,GRID_SIZE):
        if ocean[x,ny]==1: break
        d+=1
    readings["UP"]=d
    # DOWN
    d=0
    for ny in range(y-1,-1,-1):
        if ocean[x,ny]==1: break
        d+=1
    readings["DOWN"]=d
    # RIGHT
    d=0
    for nx in range(x+1,GRID_SIZE):
        if ocean[nx,y]==1: break
        d+=1
    readings["RIGHT"]=d
    # LEFT
    d=0
    for nx in range(x-1,-1,-1):
        if ocean[nx,y]==1: break
        d+=1
    readings["LEFT"]=d
    return readings

# ---------- Dynamic Obstacles ----------
def move_obstacles():
    global dynamic_obstacles, ocean
    # clear old positions
    for x,y in dynamic_obstacles:
        ocean[x,y] = 0
    new_positions = []
    for (x,y) in dynamic_obstacles:
        dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
        nx, ny = x+dx, y+dy
        # keep inside grid & avoid sub/goal
        if (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE 
            and (nx,ny) != tuple(sub_pos) and (nx,ny) != tuple(goal_pos)):
            new_positions.append((nx,ny))
        else:
            new_positions.append((x,y))  # stay if invalid
    # mark new positions
    for x,y in new_positions:
        ocean[x,y] = 1
    dynamic_obstacles = new_positions

# ---------- Update ----------
def update(frame):
    global sub_pos, slam_map, planned_path

    # Move dynamic obstacles
    move_obstacles()

    # Update SLAM map with sonar
    x,y=sub_pos
    readings=sonar_readings(sub_pos)
    for dir,dist in readings.items():
        dx,dy=moves[dir]
        for i in range(1,dist+1):
            nx,ny=x+dx*i,y+dy*i
            if 0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE:
                slam_map[nx,ny]=0
        nx,ny=x+dx*(dist+1),y+dy*(dist+1)
        if 0<=nx<GRID_SIZE and 0<=ny<GRID_SIZE:
            slam_map[nx,ny]=1

    # If no planned path or blocked, replan
    if not planned_path or planned_path[0]!=tuple(sub_pos):
        planned_path = astar(ocean, tuple(sub_pos), tuple(goal_pos))

    # Follow path
    if len(planned_path)>1:
        planned_path.pop(0)  # remove current
        next_step=planned_path[0]
        sub_pos[0],sub_pos[1]=next_step
        path.append(tuple(sub_pos))

    # --- Plot ---
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(ocean.T,cmap="Blues",origin="lower")
    px,py=zip(*path)
    plt.plot(px,py,color="red")
    plt.scatter(sub_pos[0],sub_pos[1],color="yellow",s=100)
    plt.scatter(goal_pos[0],goal_pos[1],color="green",s=100,marker="X")
    plt.title("Real Ocean Grid (Dynamic Obstacles)")

    plt.subplot(1,2,2)
    plt.imshow(slam_map.T,cmap="gray",origin="lower")
    plt.scatter(sub_pos[0],sub_pos[1],color="yellow",s=100)
    plt.scatter(goal_pos[0],goal_pos[1],color="green",s=100,marker="X")
    plt.title("SLAM Map (Estimated)")

# ---------- Run ----------
fig=plt.figure(figsize=(10,5))
ani=animation.FuncAnimation(fig, update, frames=200, interval=300, repeat=False)
plt.show()
