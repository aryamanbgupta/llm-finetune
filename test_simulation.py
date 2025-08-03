#!/usr/bin/env python3
"""
Test script to diagnose cricket simulation issues
"""

import sys
import logging
from pathlib import Path

# Configure logging to see all messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test imports
try:
    from simulator_v1 import (
        OUTCOMES, OUTCOME2TOK, Player, MatchState,
        create_sample_teams, predict_outcome, load_model, simulate_match
    )
    print("✓ Successfully imported from simulator_v1")
except Exception as e:
    print(f"✗ Failed to import from simulator_v1: {e}")
    sys.exit(1)

def test_basic_simulation():
    """Test a single match simulation step by step"""
    print("\n" + "="*60)
    print("TESTING BASIC SIMULATION")
    print("="*60)
    
    # Create teams
    team1_players, team2_players = create_sample_teams()
    print(f"✓ Created teams with {len(team1_players)} players each")
    
    # Create match state
    match = MatchState("India", "Sri Lanka", team1_players, team2_players)
    match.initialize_innings()
    print(f"✓ Initialized match state")
    print(f"  Batting: {match.batting_team}")
    print(f"  Bowling: {match.bowling_team}")
    print(f"  Batsmen: {match.current_batsmen[0].name}, {match.current_batsmen[1].name}")
    print(f"  Bowler: {match.current_bowler.name}")
    
    # Test prompt generation
    try:
        prompt = match.generate_prompt()
        print(f"\n✓ Generated prompt:")
        print(f"  {prompt}")
    except Exception as e:
        print(f"\n✗ Failed to generate prompt: {e}")
        return False
    
    # Test 10 balls
    print(f"\nSimulating 10 balls...")
    for i in range(10):
        try:
            # Generate prompt
            prompt = match.generate_prompt()
            
            # Simulate outcome (without model)
            import random
            outcome = random.choice(OUTCOMES)
            
            print(f"\nBall {i+1}:")
            print(f"  Score: {match.score}/{match.wickets}")
            print(f"  Outcome: {outcome}")
            
            # Process ball
            match.process_ball(outcome)
            
            # Check for all out
            if match.wickets >= 10:
                print(f"  ALL OUT!")
                break
                
        except Exception as e:
            print(f"✗ Error on ball {i+1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n✓ Successfully simulated {i+1} balls")
    print(f"  Final score: {match.score}/{match.wickets}")
    return True

def test_model_loading(model_path: str):
    """Test model loading and prediction"""
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    try:
        print(f"Loading model from: {model_path}")
        model, tokenizer, device = load_model(model_path)
        print(f"✓ Model loaded successfully on {device}")
        
        # Test prediction
        test_prompt = "0.0: 0/0 | Recent:  | P:0@0b(0.0rr) | Bumrah(Econ0.0) vs Rohit(new,0 Runs @ 0 SR) | PP 0.0 | India vs Sri Lanka, GEN"
        outcome, prob_dict = predict_outcome(test_prompt, model, tokenizer, device)
        
        print(f"\n✓ Test prediction successful:")
        print(f"  Prompt: {test_prompt[:50]}...")
        print(f"  Outcome: {outcome}")
        print(f"  Probabilities:")
        for o, p in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"    {o}: {p:.3f}")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_match(model_path: str):
    """Test a complete match simulation"""
    print("\n" + "="*60)
    print("TESTING FULL MATCH SIMULATION")
    print("="*60)
    
    try:
        # Load model
        model, tokenizer, device = load_model(model_path)
        
        # Create teams
        team1_players, team2_players = create_sample_teams()
        
        # Simulate match
        print("Starting match simulation...")
        result = simulate_match(
            model, tokenizer, device,
            "India", "Sri Lanka",
            team1_players, team2_players,
            "Test Stadium",
            verbose=False
        )
        
        print(f"\n✓ Match completed successfully!")
        print(f"  Winner: {result['winner']} {result['margin']}")
        print(f"  Scores:")
        for team, score in result['innings_scores'].items():
            print(f"    {team}: {score['runs']}/{score['wickets']} ({score['overs']} overs)")
        
        return True
        
    except Exception as e:
        print(f"✗ Match simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="qwen-cricket-peft-2")
    parser.add_argument("--skip_model", action="store_true", help="Skip model loading tests")
    args = parser.parse_args()
    
    print("CRICKET SIMULATION DIAGNOSTIC TEST")
    print("="*60)
    
    # Test 1: Basic simulation
    if not test_basic_simulation():
        print("\n❌ Basic simulation test failed!")
        return
    
    if args.skip_model:
        print("\nSkipping model tests (--skip_model flag)")
        return
    
    # Test 2: Model loading
    if not test_model_loading(args.model_path):
        print("\n❌ Model loading test failed!")
        return
    
    # Test 3: Full match
    if not test_full_match(args.model_path):
        print("\n❌ Full match test failed!")
        return
    
    print("\n✅ All tests passed! The simulation system is working correctly.")
    print("\nIf parallel simulation is still failing, check:")
    print("1. Are all dependencies installed?")
    print("2. Is multiprocessing working? (try with --num_workers 1)")
    print("3. Check the full logs with --log_level DEBUG")

if __name__ == "__main__":
    main()