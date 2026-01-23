"""
Prompt Engineering for AI Graph Search
Demonstrates various prompting strategies for teaching LLMs to solve pathfinding problems
"""

import numpy as np
from collections import deque
import heapq
from typing import List, Tuple, Set, Optional

# ============================================================================
# SECTION 1: GEOGRAPHIC MAP ENVIRONMENT
# ============================================================================

class GeographicMap:
    """Simple geographic grid map with terrain obstacles"""
    
    def __init__(self, size: int = 6):
        self.size = size
        self.start = (0, 0)
        self.goal = (5, 5)
        
        # Obstacles represent mountains, water bodies, etc.
        self.obstacles = {
            (1, 1), (1, 2), (1, 3),  # Mountain range
            (3, 2), (3, 3),          # Lake
            (4, 4)                   # Building
        }
        
    def visualize(self) -> str:
        """Visualize the map"""
        vis = "Geographic Map:\n"
        vis += "-" * (self.size * 2 + 1) + "\n"
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.start:
                    vis += "S "
                elif (i, j) == self.goal:
                    vis += "G "
                elif (i, j) in self.obstacles:
                    vis += "X "
                else:
                    vis += ". "
            vis += "\n"
        vis += "-" * (self.size * 2 + 1)
        return vis
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-directional movement)"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for dx, dy in directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < self.size and 
                0 <= new_pos[1] < self.size and 
                new_pos not in self.obstacles):
                neighbors.append(new_pos)
        return neighbors
    
    def get_description(self) -> str:
        """Natural language description of the map"""
        return f"""
Geographic Map Details:
- Map size: {self.size}x{self.size} grid
- Start location: {self.start} (Northwest corner)
- Goal location: {self.goal} (Southeast corner)
- Obstacles: {sorted(self.obstacles)}
- Movement: Can move North, South, East, West (one cell at a time)
- Task: Find path from start to goal avoiding obstacles
"""


# ============================================================================
# SECTION 2: GROUND TRUTH SEARCH ALGORITHMS
# ============================================================================

def bfs_search(map_env: GeographicMap) -> Tuple[List[Tuple[int, int]], int]:
    """Breadth-First Search - finds shortest path"""
    queue = deque([(map_env.start, [map_env.start])])
    visited = {map_env.start}
    nodes_explored = 0
    
    while queue:
        pos, path = queue.popleft()
        nodes_explored += 1
        
        if pos == map_env.goal:
            return path, nodes_explored
        
        for neighbor in map_env.get_neighbors(pos):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return [], nodes_explored


def dfs_search(map_env: GeographicMap) -> Tuple[List[Tuple[int, int]], int]:
    """Depth-First Search - finds a path (not necessarily shortest)"""
    stack = [(map_env.start, [map_env.start])]
    visited = {map_env.start}
    nodes_explored = 0
    
    while stack:
        pos, path = stack.pop()
        nodes_explored += 1
        
        if pos == map_env.goal:
            return path, nodes_explored
        
        for neighbor in map_env.get_neighbors(pos):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
    return [], nodes_explored


def manhattan_distance(pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """Heuristic for A* search"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar_search(map_env: GeographicMap) -> Tuple[List[Tuple[int, int]], int]:
    """A* Search - optimal path with heuristic guidance"""
    open_set = [(0, map_env.start)]
    came_from = {}
    g_score = {map_env.start: 0}
    nodes_explored = 0
    
    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        
        if current == map_env.goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(map_env.start)
            return path[::-1], nodes_explored
        
        for neighbor in map_env.get_neighbors(current):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, map_env.goal)
                heapq.heappush(open_set, (f_score, neighbor))
    
    return [], nodes_explored


def run_all_algorithms(map_env: GeographicMap):
    """Compare all search algorithms"""
    print("\n" + "=" * 70)
    print("GROUND TRUTH ALGORITHM COMPARISON")
    print("=" * 70)
    
    bfs_path, bfs_nodes = bfs_search(map_env)
    print(f"\nBFS (Breadth-First Search):")
    print(f"  Path: {bfs_path}")
    print(f"  Path length: {len(bfs_path)}")
    print(f"  Nodes explored: {bfs_nodes}")
    
    dfs_path, dfs_nodes = dfs_search(map_env)
    print(f"\nDFS (Depth-First Search):")
    print(f"  Path: {dfs_path}")
    print(f"  Path length: {len(dfs_path)}")
    print(f"  Nodes explored: {dfs_nodes}")
    
    astar_path, astar_nodes = astar_search(map_env)
    print(f"\nA* Search (with Manhattan distance heuristic):")
    print(f"  Path: {astar_path}")
    print(f"  Path length: {len(astar_path)}")
    print(f"  Nodes explored: {astar_nodes}")
    
    return bfs_path, dfs_path, astar_path


# ============================================================================
# SECTION 3: PROMPT ENGINEERING STRATEGIES
# ============================================================================

def create_prompts(map_env: GeographicMap) -> dict:
    """Generate different prompt engineering strategies"""
    
    prompts = {}
    
    # Strategy 1: Direct/Naive Prompt
    prompts["1_direct"] = f"""
Find a path from {map_env.start} to {map_env.goal} in a {map_env.size}x{map_env.size} grid.
Obstacles are at: {list(map_env.obstacles)}
You can move up, down, left, right.

Path:
"""
    
    # Strategy 2: Structured Problem Definition
    prompts["2_structured"] = f"""
PATHFINDING PROBLEM:
- Grid size: {map_env.size}x{map_env.size}
- Start position: {map_env.start}
- Goal position: {map_env.goal}
- Obstacles: {list(map_env.obstacles)}
- Valid actions: Move to adjacent cell (North, South, East, West)
- Constraint: Cannot move through obstacles or outside grid

TASK: Find a valid path from start to goal.
OUTPUT FORMAT: List of coordinates [(x1,y1), (x2,y2), ...]

SOLUTION:
"""
    
    # Strategy 3: Visual Representation
    prompts["3_visual"] = f"""
Geographic Map (S=start, G=goal, X=obstacle, .=free):

{map_env.visualize()}

Find the shortest path from S to G.
Provide the path as a sequence of coordinates.

PATH:
"""
    
    # Strategy 4: Chain-of-Thought (CoT)
    prompts["4_cot"] = f"""
Navigate from {map_env.start} to {map_env.goal} avoiding obstacles at {list(map_env.obstacles)}.

Let's think step-by-step:
1. Starting at {map_env.start}, what positions can we move to?
2. Which of these moves avoid obstacles?
3. Continue exploring until we reach {map_env.goal}
4. Track the complete path

Step-by-step solution:
"""
    
    # Strategy 5: Few-Shot Learning
    prompts["5_few_shot"] = f"""
EXAMPLE 1:
Grid: 4x4, Start: (0,0), Goal: (3,3), Obstacles: [(1,1), (2,2)]
Solution: [(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3)]

EXAMPLE 2:
Grid: 3x3, Start: (0,0), Goal: (2,2), Obstacles: [(1,1)]
Solution: [(0,0), (1,0), (2,0), (2,1), (2,2)]

NOW SOLVE:
Grid: {map_env.size}x{map_env.size}
Start: {map_env.start}
Goal: {map_env.goal}
Obstacles: {list(map_env.obstacles)}

Solution:
"""
    
    # Strategy 6: Algorithm-Specific (BFS)
    prompts["6_bfs_algorithm"] = f"""
Use BREADTH-FIRST SEARCH algorithm to solve:

Problem Setup:
- Start: {map_env.start}
- Goal: {map_env.goal}
- Obstacles: {list(map_env.obstacles)}
- Grid: {map_env.size}x{map_env.size}

BFS Algorithm Steps:
1. Initialize queue with start position and empty path
2. Mark start as visited
3. While queue is not empty:
   a. Dequeue position and current path
   b. If position is goal, return path
   c. For each valid neighbor (up/down/left/right):
      - If not visited and not obstacle:
        * Mark as visited
        * Enqueue with updated path
4. Return path to goal

Execute this algorithm and provide the resulting path:
"""
    
    # Strategy 7: Algorithm-Specific (A*)
    prompts["7_astar_algorithm"] = f"""
Use A* SEARCH with Manhattan distance heuristic:

Problem:
- Start: {map_env.start}
- Goal: {map_env.goal}
- Obstacles: {list(map_env.obstacles)}

A* Algorithm:
- Heuristic h(n) = |x_goal - x_n| + |y_goal - y_n|
- Cost function f(n) = g(n) + h(n)
  where g(n) = steps from start to n

Find optimal path using A*:
"""
    
    # Strategy 8: Code Generation
    prompts["8_code_generation"] = f"""
Write Python code to find path using BFS:

from collections import deque

def find_path():
    start = {map_env.start}
    goal = {map_env.goal}
    obstacles = set({list(map_env.obstacles)})
    size = {map_env.size}
    
    # Implement BFS here
    # Return path as list of tuples
    
find_path()

Complete the implementation and show the path:
"""
    
    # Strategy 9: Comparative Analysis
    prompts["9_comparative"] = f"""
Compare BFS vs DFS for this pathfinding problem:

Map: {map_env.size}x{map_env.size}
Start: {map_env.start}
Goal: {map_env.goal}
Obstacles: {list(map_env.obstacles)}

Analysis:
1. BFS characteristics: Complete, optimal, explores level-by-level
2. DFS characteristics: Complete, not optimal, explores depth-first

Which algorithm is better for finding the SHORTEST path?
Provide the path using the better algorithm:
"""
    
    # Strategy 10: Constraint-Based
    prompts["10_constraints"] = f"""
Find path with these constraints:

HARD CONSTRAINTS:
- Must start at {map_env.start}
- Must end at {map_env.goal}
- Cannot pass through {list(map_env.obstacles)}
- Can only move to adjacent cells (4-directional)
- Must stay within {map_env.size}x{map_env.size} grid

SOFT CONSTRAINTS:
- Prefer shortest path
- Minimize turns if possible

Valid path satisfying all constraints:
"""
    
    return prompts


# ============================================================================
# SECTION 4: LLM QUERY FUNCTION (Local Model)
# ============================================================================

def query_llm_local(prompt: str, max_tokens: int = 300) -> str:
    """
    Query local LLM model using transformers library
    Requires: pip install transformers torch accelerate
    """
    try:
        from transformers import pipeline
        import torch
        
        # Initialize model (only once)
        if not hasattr(query_llm_local, 'pipe'):
            print("Loading local model (first time only)...")
            query_llm_local.pipe = pipeline(
                "text-generation",
                model="microsoft/phi-2",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        result = query_llm_local.pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            return_full_text=False
        )
        
        return result[0]["generated_text"]
    
    except Exception as e:
        return f"Error: {e}\n(Install: pip install transformers torch accelerate)"


# ============================================================================
# SECTION 5: MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main demonstration function"""
    
    # Create map environment
    map_env = GeographicMap()
    
    print("=" * 70)
    print("PROMPT ENGINEERING FOR GRAPH SEARCH ALGORITHMS")
    print("=" * 70)
    
    # Display map
    print(map_env.visualize())
    print(map_env.get_description())
    
    # Run ground truth algorithms
    bfs_path, dfs_path, astar_path = run_all_algorithms(map_env)
    
    # Generate prompts
    prompts = create_prompts(map_env)
    
    # Demonstrate each prompting strategy
    print("\n" + "=" * 70)
    print("PROMPT ENGINEERING STRATEGIES")
    print("=" * 70)
    
    for name, prompt in prompts.items():
        print(f"\n{'=' * 70}")
        print(f"STRATEGY: {name.replace('_', ' ').upper()}")
        print('=' * 70)
        print(f"\nPROMPT:")
        print(prompt)
        
        # Query LLM (commented out by default - students can uncomment)
        # print("\nLLM RESPONSE:")
        # response = query_llm_local(prompt)
        # print(response)
        
        print(f"\nGROUND TRUTH (BFS): {bfs_path}")
        print(f"Path length: {len(bfs_path)}")
        print("-" * 70)


# ============================================================================
# SECTION 6: INDIVIDUAL RUNNABLE BLOCKS FOR STUDENTS
# ============================================================================

def demo_block_1_basic_map():
    """Block 1: Understanding the map environment"""
    print("\n=== BLOCK 1: Map Environment ===")
    map_env = GeographicMap()
    print(map_env.visualize())
    print(map_env.get_description())
    print(f"Neighbors of (0,0): {map_env.get_neighbors((0,0))}")
    print(f"Neighbors of (1,1): {map_env.get_neighbors((1,1))} - obstacle!")


def demo_block_2_bfs():
    """Block 2: BFS algorithm"""
    print("\n=== BLOCK 2: BFS Search ===")
    map_env = GeographicMap()
    path, nodes = bfs_search(map_env)
    print(f"BFS Path: {path}")
    print(f"Length: {len(path)}, Nodes explored: {nodes}")


def demo_block_3_dfs():
    """Block 3: DFS algorithm"""
    print("\n=== BLOCK 3: DFS Search ===")
    map_env = GeographicMap()
    path, nodes = dfs_search(map_env)
    print(f"DFS Path: {path}")
    print(f"Length: {len(path)}, Nodes explored: {nodes}")


def demo_block_4_astar():
    """Block 4: A* algorithm"""
    print("\n=== BLOCK 4: A* Search ===")
    map_env = GeographicMap()
    path, nodes = astar_search(map_env)
    print(f"A* Path: {path}")
    print(f"Length: {len(path)}, Nodes explored: {nodes}")


def demo_block_5_prompt_comparison():
    """Block 5: Compare different prompts"""
    print("\n=== BLOCK 5: Prompt Strategy Comparison ===")
    map_env = GeographicMap()
    prompts = create_prompts(map_env)
    
    # Show just a few strategies
    for name in ["1_direct", "3_visual", "6_bfs_algorithm"]:
        print(f"\n--- {name} ---")
        print(prompts[name][:200] + "...")


def demo_block_6_llm_query():
    """Block 6: Query LLM with a prompt"""
    print("\n=== BLOCK 6: LLM Query ===")
    map_env = GeographicMap()
    prompts = create_prompts(map_env)
    
    # Try the visual prompt
    prompt = prompts["3_visual"]
    print("Querying LLM...")
    print(f"Prompt:\n{prompt}")
    
    response = query_llm_local(prompt)
    print(f"\nLLM Response:\n{response}")


# ============================================================================
# RUN DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    # Uncomment the sections you want to run:
    
    # Full demonstration
    main()
    
    # Individual blocks (for students to run separately)
    # demo_block_1_basic_map()
    # demo_block_2_bfs()
    # demo_block_3_dfs()
    # demo_block_4_astar()
    # demo_block_5_prompt_comparison()
    # demo_block_6_llm_query()
