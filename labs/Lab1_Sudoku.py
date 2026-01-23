import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# ============ CONFIG ============
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
# Use a better instruction-following model
MODEL_NAME = "microsoft/phi-2"  # or try "HuggingFaceH4/zephyr-7b-beta"

# ============ BASELINE WITH STEPS ============
class SudokuSolver:
    def __init__(self, grid):
        self.grid = grid.copy()
        self.steps = []
        
    def is_valid(self, row, col, num):
        if num in self.grid[row]:
            return False
        if num in self.grid[:, col]:
            return False
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in self.grid[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True
    
    def solve(self):
        empty = np.argwhere(self.grid == 0)
        if len(empty) == 0:
            return True
        
        row, col = empty[0]
        
        for num in range(1, 10):
            if self.is_valid(row, col, num):
                self.grid[row, col] = num
                self.steps.append(f"Placed {num} at ({row},{col}) - valid by constraints")
                
                if self.solve():
                    return True
                
                self.grid[row, col] = 0
                self.steps.append(f"Backtracked from {num} at ({row},{col})")
        
        return False
    
    def print_steps(self, max_steps=15):
        print(f"Total steps: {len(self.steps)}")
        print(f"\nFirst {max_steps} steps:")
        for i, step in enumerate(self.steps[:max_steps], 1):
            print(f"  {i}. {step}")

# ============ LLM AGENT WITH BETTER PROMPTS ============
class SudokuLLMAgent:
    def __init__(self, model_name=MODEL_NAME, token=HF_TOKEN):
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=token,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model loaded!")
        
    def grid_to_string(self, grid):
        lines = []
        for i in range(9):
            row = " ".join(str(grid[i, j]) for j in range(9))
            lines.append(row)
        return "\n".join(lines)
    
    def generate_response(self, prompt, max_tokens=150):
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,  # Prevent repetition
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new content after the prompt
        response = response[len(prompt):].strip()
        # Take only first few lines to avoid rambling
        lines = response.split('\n')
        return '\n'.join(lines[:5])
    
    def get_candidates(self, grid, row, col):
        """Get valid candidates for a cell"""
        if grid[row, col] != 0:
            return []
        
        candidates = set(range(1, 10))
        candidates -= set(grid[row])
        candidates -= set(grid[:, col])
        
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_vals = grid[box_row:box_row+3, box_col:box_col+3].flatten()
        candidates -= set(box_vals)
        
        return sorted(list(candidates))
    
    def solve_cot_iterative(self, grid, max_steps=5):
        """Solve step-by-step with constrained reasoning"""
        current_grid = grid.copy()
        steps = []
        
        print("\n=== Chain-of-Thought Step-by-Step ===")
        
        for step_num in range(max_steps):
            empty_cells = np.argwhere(current_grid == 0)
            if len(empty_cells) == 0:
                print("\nPuzzle solved!")
                break
            
            row, col = empty_cells[0]
            candidates = self.get_candidates(current_grid, row, col)
            
            if not candidates:
                print(f"No valid candidates for ({row},{col}) - stopping")
                break
            
            # Use actual constraint info in prompt
            row_nums = [x for x in current_grid[row] if x != 0]
            col_nums = [x for x in current_grid[:, col] if x != 0]
            
            #For students to do: Try different kinds of prompts here to see how CoT responses differ
            #You will be amazed -- if you pick microsoft's phi2 (generates garbage!)
            prompt = f"""Question: In Sudoku, cell ({row},{col}) is empty. 
Row {row} contains: {row_nums}
Column {col} contains: {col_nums}
Valid candidates are: {candidates}

Which number should go in cell ({row},{col})? Pick one number from the candidates.

Answer: The number is"""
            
            print(f"\n--- Step {step_num + 1}: Cell ({row},{col}) ---")
            print(f"Candidates: {candidates}")
            
            response = self.generate_response(prompt, max_tokens=50)
            print(f"LLM says: {response}")
            
            # Parse the response - look for a digit
            import re
            numbers = re.findall(r'\b([1-9])\b', response)
            
            if numbers:
                value = int(numbers[0])
                if value in candidates:
                    current_grid[row, col] = value
                    steps.append({
                        'step': step_num + 1,
                        'cell': (row, col),
                        'value': value,
                        'candidates': candidates,
                        'response': response[:100],
                        'status': 'valid'
                    })
                    print(f"✓ Placed {value}")
                else:
                    steps.append({
                        'step': step_num + 1,
                        'cell': (row, col),
                        'value': value,
                        'candidates': candidates,
                        'response': response[:100],
                        'status': f'invalid - {value} not in candidates'
                    })
                    print(f"✗ Invalid: {value} not in {candidates}")
                    break
            else:
                print(f"✗ Could not parse number from response")
                break
        
        return current_grid, steps
    
    def solve_direct_verbose(self, grid):
        """Ask for just the next number"""
        empty = np.argwhere(grid == 0)
        if len(empty) == 0:
            return "Already solved"
        
        row, col = empty[0]
        candidates = self.get_candidates(grid, row, col)
        
        prompt = f"""Sudoku puzzle. First empty cell is ({row},{col}).
Valid options: {candidates}

Pick one number: """
        
        print("\n=== Direct Solution (First Cell) ===")
        response = self.generate_response(prompt, max_tokens=30)
        print(f"LLM response: {response}")
        
        return response

# ============ EVALUATION ============
def main():
    # Easier puzzle for testing
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    print("=" * 60)
    print("APPROACH 1: BASELINE (Backtracking)")
    print("=" * 60)
    solver = SudokuSolver(puzzle)
    start = time.time()
    success = solver.solve()
    baseline_time = time.time() - start
    
    print(f"Result: {'Solved' if success else 'Failed'}")
    print(f"Time: {baseline_time:.3f}s")
    solver.print_steps(max_steps=15)
    
    print("\n" + "=" * 60)
    print("APPROACH 2: LLM Chain-of-Thought")
    print("=" * 60)
    agent = SudokuLLMAgent()
    
    start = time.time()
    result_grid, cot_steps = agent.solve_cot_iterative(puzzle, max_steps=5)
    cot_time = time.time() - start
    
    print(f"\n\nSummary:")
    print(f"Time: {cot_time:.3f}s")
    print(f"Valid steps: {sum(1 for s in cot_steps if s['status'] == 'valid')}")
    
    print("\n" + "=" * 60)
    print("APPROACH 3: LLM Direct")
    print("=" * 60)
    
    start = time.time()
    direct_result = agent.solve_direct_verbose(puzzle)
    direct_time = time.time() - start
    print(f"Time: {direct_time:.3f}s")
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Baseline: {baseline_time:.3f}s, {len(solver.steps)} steps")
    print(f"CoT: {cot_time:.3f}s, {len(cot_steps)} LLM calls")
    print(f"Direct: {direct_time:.3f}s, 1 LLM call")

if __name__ == "__main__":
    main()
