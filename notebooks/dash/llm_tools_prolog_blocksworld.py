"""
Comparison: Pure LLM vs LLM+ReAct+Prolog for Blocks World
Demonstrates why hybrid systems are necessary
Developed for: AI course @ BITS Pilani, Goa campus
"""

import subprocess
import tempfile
import os
import re
from typing import Dict, List, Optional

# PROLOG PLANNER (Copied from the Book: Bratko's Prolog 4ed. Ch 17, Fig. 17.6)
BRATKO_PROLOG_PLANNER = """
can(move(Block, From, To), [clear(Block), clear(To), on(Block, From)]) :-
    block(Block),
    object(To),
    To \\== Block,
    object(From),
    From \\== To,
    Block \\== From.

adds(move(X, From, To), [on(X, To), clear(From)]).
deletes(move(X, From, To), [on(X, From), clear(To)]).

object(X) :- place(X).
object(X) :- block(X).

block(a). block(b). block(c).
place(1). place(2). place(3). place(4).

state1([clear(2), clear(4), clear(b), clear(c), on(a,1), on(b,3), on(c,a)]).

plan(State, Goals, []) :-
    satisfied(State, Goals).

plan(State, Goals, Plan) :-
    conc(PrePlan, [Action], Plan),
    select(State, Goals, Goal),
    achieves(Action, Goal),
    can(Action, Condition),
    preserves(Action, Goals),
    regress(Goals, Action, RegressedGoals),
    plan(State, RegressedGoals, PrePlan).

satisfied(State, Goals) :-
    delete_all(Goals, State, []).

select(State, Goals, Goal) :-
    member(Goal, Goals).

achieves(Action, Goal) :-
    adds(Action, Goals),
    member(Goal, Goals).

preserves(Action, Goals) :-
    deletes(Action, Relations),
    \\+ (member(Goal, Relations), member(Goal, Goals)).

regress(Goals, Action, RegressedGoals) :-
    adds(Action, NewRelations),
    delete_all(Goals, NewRelations, RestGoals),
    can(Action, Condition),
    addnew(Condition, RestGoals, RegressedGoals).

addnew([], L, L).
addnew([Goal | _], Goals, _) :-
    impossible(Goal, Goals), !, fail.
addnew([X | L1], L2, L3) :-
    member(X, L2), !,
    addnew(L1, L2, L3).
addnew([X | L1], L2, [X | L3]) :-
    addnew(L1, L2, L3).

impossible(on(X, X), _).
impossible(on(X, Y), Goals) :-
    member(clear(Y), Goals).
impossible(on(X, Y1), Goals) :-
    member(on(X, Y2), Goals), Y1 \\== Y2.
impossible(on(X1, Y), Goals) :-
    member(on(X2, Y), Goals), X1 \\== X2.
impossible(clear(X), Goals) :-
    member(on(_, X), Goals).

delete_all([], _, []).
delete_all([X | L1], L2, Diff) :-
    member(X, L2), !,
    delete_all(L1, L2, Diff).
delete_all([X | L1], L2, [X | Diff]) :-
    delete_all(L1, L2, Diff).

conc([], L, L).
conc([X | L1], L2, [X | L3]) :-
    conc(L1, L2, L3).
"""

class PrologPlanner:
    """Interface to Prolog planner."""
    
    def __init__(self):
        self.prolog_code = BRATKO_PROLOG_PLANNER
        self.temp_file = None
    
    def plan(self, initial_state: str, goals: str) -> Dict:
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pl', delete=False
        )
        self.temp_file.write(self.prolog_code)
        self.temp_file.close()
        
        try:
            query = f"plan({initial_state}, {goals}, Plan), write(Plan), nl, halt."
            
            result = subprocess.run(
                ['swipl', '-q', '-s', self.temp_file.name, '-g', query, '-t', 'halt(1)'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                plan_str = result.stdout.strip()
                plan = self._parse_plan(plan_str)
                return {
                    'success': True,
                    'plan': plan,
                    'raw': plan_str
                }
            else:
                return {
                    'success': False,
                    'error': 'No plan found or query failed',
                    'stderr': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout'}
        except FileNotFoundError:
            return {'success': False, 'error': 'SWI-Prolog not installed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            if self.temp_file:
                try:
                    os.unlink(self.temp_file.name)
                except:
                    pass
    
    def _parse_plan(self, plan_str: str) -> List[str]:
        moves = re.findall(r'move\([^)]+\)', plan_str)
        return moves

# PURE LLM AGENT (No ReAct, No Tools)
class PureLLMAgent:
    """
    Pure LLM agent - no tools, no structured reasoning.
    Just asks LLM to solve the problem directly.
    """
    
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        print(f"Loading model: {model_name}...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Model loaded on {self.device}")
            
        except ImportError:
            print("ERROR: transformers not installed")
            raise
    
    def solve_blocks_world_direct(self, problem: str) -> str:
        """
        Ask LLM to solve blocks world directly.
        No tools, no structured reasoning.
        """
        print("\n" + "="*70)
        print("PURE LLM (NO TOOLS, NO REACT)")
        print("="*70 + "\n")
        
        prompt = f"""You are a planning assistant. Solve this blocks world problem.

Problem:
{problem}

Provide a sequence of moves in the format: move(block, from, to)

Your answer:"""

        print(f"Prompt: {prompt}")        
        print("[Prompt sent to LLM]\n")
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        print("[LLM Response]")
        print(response)
        print()
        
        return response

# REACT + PROLOG AGENT (from previous code)
class ReactPrologAgent:
    """LLM Agent with ReAct framework and Prolog tool."""
    
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.prolog_tool = PrologPlanner()
    
    def solve_blocks_world_react(self, problem: str) -> str:
        print("\n" + "="*70)
        print("REACT + PROLOG TOOL")
        print("="*70 + "\n")
        
        # Default to Bratko's example
        initial = "[clear(2), clear(4), clear(b), clear(c), on(a,1), on(b,3), on(c,a)]"
        goals = "[on(a,b), on(b,c)]"
        
        # THOUGHT 1
        thought1 = "I need a verified plan. Use Prolog planner tool."
        print(f"[Thought 1] {thought1}\n")
        
        # ACTION 1
        action1 = f"prolog_plan(initial, goals)"
        print(f"[Action 1] {action1}\n")
        
        # Execute Prolog tool
        result = self.prolog_tool.plan(initial, goals)
        
        # OBSERVATION 1
        if result['success']:
            obs1 = f"Prolog planner found plan: {result['plan']}"
            print(f"[Observation 1] {obs1}\n")
            
            # THOUGHT 2
            thought2 = "Plan is verified correct. Present to user."
            print(f"[Thought 2] {thought2}\n")
            
            # ACTION 2
            action2 = "finish"
            print(f"[Action 2] {action2}\n")
            
            # Format answer
            output = "Plan (verified by Prolog):\n"
            for i, move in enumerate(result['plan'], 1):
                output += f"  {i}. {move}\n"
            
            return output
        else:
            return f"Failed: {result['error']}"

# PLAN VALIDATOR
class PlanValidator:
    """Validate if a plan is correct."""
    
    @staticmethod
    def validate_plan(plan_text: str) -> Dict:
        """
        Extract moves from LLM output and check if they're valid.
        
        Returns dict with:
        - moves: List of extracted moves
        - valid: Whether plan appears valid
        - issues: List of problems found
        """
        # Extract move(...) patterns
        moves = re.findall(r'move\(([^,]+),\s*([^,]+),\s*([^)]+)\)', plan_text)
        
        issues = []
        
        # Initial state: c on a, a on 1, b on 3
        state = {
            'on': {'c': 'a', 'a': '1', 'b': '3'},
            'clear': {'c', 'b', '2', '4'}
        }
        
        for i, (block, from_loc, to_loc) in enumerate(moves, 1):
            block = block.strip()
            from_loc = from_loc.strip()
            to_loc = to_loc.strip()
            
            # Check if block is clear
            if block not in state['clear']:
                issues.append(f"Move {i}: {block} is not clear (has something on top)")
            
            # Check if block is actually on from_loc
            if block not in state['on'] or state['on'][block] != from_loc:
                issues.append(f"Move {i}: {block} is not on {from_loc}")
            
            # Check if to_loc is clear
            if to_loc not in state['clear']:
                issues.append(f"Move {i}: {to_loc} is not clear")
            
            # Apply move (simplified - doesn't handle all edge cases)
            if block in state['on']:
                old_loc = state['on'][block]
                state['clear'].add(old_loc)
            
            state['on'][block] = to_loc
            state['clear'].discard(to_loc)
            state['clear'].add(block)
        
        # Check if goal is achieved: a on b, b on c
        goal_achieved = (
            state['on'].get('a') == 'b' and
            state['on'].get('b') == 'c'
        )
        
        return {
            'moves': [f"move({b},{f},{t})" for b, f, t in moves],
            'num_moves': len(moves),
            'valid': len(issues) == 0,
            'issues': issues,
            'goal_achieved': goal_achieved
        }


def run_comparison():
    problem = """
Initial configuration:
c
a   b
=========
1 2 3 4

Where:
- Block c is on block a
- Block a is on place 1
- Block b is on place 3
- Places 2 and 4 are empty
- Blocks b and c have nothing on top (are clear)

Goal: Stack the blocks so that a is on b, and b is on c
"""
    
    print(problem)
    
    # Check if Prolog is available
    prolog_available = True
    try:
        subprocess.run(['swipl', '--version'], capture_output=True, check=True)
    except:
        prolog_available = False
        print("\n[WARNING] SWI-Prolog not found. Skipping ReAct+Prolog comparison.\n")
    
    # Test 1: Pure LLM
    print("\n\nAPPROACH 1: PURE LLM")
    
    try:
        pure_agent = PureLLMAgent(model_name="microsoft/Phi-3-mini-4k-instruct")
        pure_result = pure_agent.solve_blocks_world_direct(problem)
        
        # Validate the plan
        validation = PlanValidator.validate_plan(pure_result)
        
        print("\n[VALIDATION]")
        print(f"Extracted moves: {validation['moves']}")
        print(f"Number of moves: {validation['num_moves']}")
        print(f"Valid: {validation['valid']}")
        print(f"Goal achieved: {validation['goal_achieved']}")
        
        if validation['issues']:
            print(f"\nISSUES FOUND:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load LLM: {e}")
        return
    
    # Test 2: ReAct + Prolog
    if prolog_available:
        print("\n\nAPPROACH 2: REACT + PROLOG TOOL")
        
        react_agent = ReactPrologAgent()
        react_result = react_agent.solve_blocks_world_react(problem)
        
        print(react_result)
        
        # Validate Prolog plan
        validation = PlanValidator.validate_plan(react_result)
        
        print("[VALIDATION]")
        print(f"Number of moves: {validation['num_moves']}")
        print(f"Valid: {validation['valid']}")
        print(f"Goal achieved: {validation['goal_achieved']}")
        
        if validation['issues']:
            print(f"\nISSUES FOUND:")
            for issue in validation['issues']:
                print(f"  - {issue}")
    

if __name__ == "__main__":
    run_comparison()

