#!/usr/bin/env python3
"""
Series Simulation Wrapper - Simple Stable Version
Uses float32 + eager attention for maximum stability (like Mac Mini)
"""

import time
import sys
from pathlib import Path
import argparse
import logging

# Import required modules
from aus_wi_t20i_series_setup import get_all_matches, get_match_by_number
from parallel_sim_v1 import SimulationConfig, run_parallel_simulation

def create_match_teams_function(match_config):
    """Create a function that returns teams for a specific match"""
    def get_match_teams():
        return match_config['team1_players'], match_config['team2_players']
    return get_match_teams

def patch_parallel_sim_for_stability():
    """Patch parallel_sim_v1 to use stable configuration"""
    import parallel_sim_v1
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from parallel_sim_v1 import OUTCOME2TOK
    
    def load_model_for_inference_stable(model_path: str):
        """Load model with stable configuration (matches Mac Mini behavior)"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {device} with STABLE float32 configuration...")
        
        base_model_name = "Qwen/Qwen1.5-1.8B"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Add special tokens
        special_tokens = list(OUTCOME2TOK.values())
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # STABLE FIX: Use float32 + eager attention (like Mac Mini)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,  # STABLE: Use float32 like Mac Mini
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=False,
            attn_implementation="eager"  # STABLE: Use eager attention (no Flash Attention)
        )
        
        # Resize embeddings
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        # STABLE: No torch.compile for maximum stability
        print("✅ Using stable configuration: float32 + eager attention (no compilation)")
        
        return model, tokenizer, device
    
    # Patch the function
    parallel_sim_v1.load_model_for_inference = load_model_for_inference_stable
    print("✅ Patched parallel_sim_v1 for stability")

def simulate_match_series(model_path: str, simulations_per_match: int = 10000, 
                         base_output_dir: str = "series_simulation_results",
                         batch_size: int = 64,  # Conservative for stability
                         num_workers: int = 16,  # Conservative for stability  
                         prompt_queue_size: int = 2048,
                         batch_queue_size: int = 32,
                         result_queue_size: int = 1024,
                         queue_timeout: float = 2.0,
                         batch_timeout: float = 0.5,
                         checkpoint_interval: int = None):
    """
    Simulate all 5 matches from AUS vs WI series with stable configuration
    """
    
    print("="*80)
    print("AUSTRALIA vs WEST INDIES T20I SERIES SIMULATION - STABLE VERSION")
    print("="*80)
    print(f"Simulations per match: {simulations_per_match:,}")
    print(f"Total simulations: {simulations_per_match * 5:,}")
    print(f"Output directory: {base_output_dir}")
    print()
    
    # Patch for stability BEFORE creating config
    patch_parallel_sim_for_stability()
    
    # Get all match configurations
    all_matches = get_all_matches()
    base_path = Path(base_output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Conservative configuration for stability
    if checkpoint_interval is None:
        checkpoint_interval = max(50, simulations_per_match // 20)
    
    config = SimulationConfig(
        num_matches=simulations_per_match,
        batch_size=batch_size,
        num_workers=num_workers,
        prompt_queue_size=prompt_queue_size,
        batch_queue_size=batch_queue_size,
        result_queue_size=result_queue_size,
        save_ball_by_ball=False,
        checkpoint_interval=checkpoint_interval,
        verbose=True,
        queue_timeout=queue_timeout,
        batch_timeout=batch_timeout
    )
    
    print(f"Stable Configuration:")
    print(f"  Batch size: {config.batch_size} (conservative)")
    print(f"  Workers: {config.num_workers} (conservative)")
    print(f"  Prompt queue: {config.prompt_queue_size:,}")
    print(f"  Result queue: {config.result_queue_size:,}")
    print(f"  Checkpoint interval: {config.checkpoint_interval}")
    print(f"  Precision: float32 + eager attention")
    print()
    
    series_start_time = time.time()
    
    # Simulate each match
    for match_num, match_config in enumerate(all_matches, 1):
        print(f"\n{'='*60}")
        print(f"SIMULATING MATCH {match_num}: {match_config['match_name']}")
        print(f"{'='*60}")
        print(f"Date: {match_config['date']}")
        print(f"Venue: {match_config['venue']}, {match_config['location']}")
        print(f"Teams: {match_config['team1_name']} vs {match_config['team2_name']}")
        print(f"Special: {match_config['special_notes']}")
        print()
        
        # Create output directory for this match
        match_output = base_path / f"match_{match_num}_{match_config['match_name'].lower().replace(' ', '_')}"
        match_output.mkdir(exist_ok=True)
        
        # MONKEY PATCH: Temporarily replace team creation function
        import parallel_sim_v1
        
        # Store original function
        original_create_teams = getattr(parallel_sim_v1, 'create_sample_teams', None)
        
        # Create new function that returns our match teams
        match_teams_func = create_match_teams_function(match_config)
        
        # Replace the function in the module
        parallel_sim_v1.create_sample_teams = match_teams_func
        
        try:
            # Run simulation with match-specific teams
            match_start_time = time.time()
            
            # Create unique output path with timestamp
            simulation_output = match_output / f"sim_{simulations_per_match}_{int(time.time())}"
            
            run_parallel_simulation(model_path, config, simulation_output)
            
            match_elapsed = time.time() - match_start_time
            print(f"\n✅ Match {match_num} completed in {match_elapsed:.1f} seconds")
            print(f"   Rate: {simulations_per_match / match_elapsed:.1f} simulations/sec")
            
            # Validate results
            checkpoints = list(simulation_output.glob("checkpoint_*"))
            if checkpoints:
                print(f"✅ Created {len(checkpoints)} checkpoints")
            else:
                print(f"❌ No checkpoints created for match {match_num}")
                
        except Exception as e:
            print(f"❌ Match {match_num} failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # RESTORE: Put back original function
            if original_create_teams:
                parallel_sim_v1.create_sample_teams = original_create_teams
    
    # Series summary
    total_elapsed = time.time() - series_start_time
    total_sims = simulations_per_match * 5
    print(f"\n{'='*80}")
    print("SERIES SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"Average per match: {total_elapsed/5:.1f} seconds")
    print(f"Total simulations: {total_sims:,}")
    print(f"Overall rate: {total_sims / total_elapsed:.1f} simulations/sec")
    print(f"\nResults saved in: {base_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate AUS vs WI T20I series - Stable Version")
    
    # Basic options
    parser.add_argument("--model_path", default="models/checkpoint-24000", 
                       help="Path to fine-tuned model")
    parser.add_argument("--simulations_per_match", type=int, default=10000,
                       help="Number of simulations per match")
    parser.add_argument("--output_dir", default="series_simulation_results_stable",
                       help="Base output directory")
    parser.add_argument("--single_match", type=int, choices=[1,2,3,4,5],
                       help="Simulate only a specific match (1-5)")
    parser.add_argument("--log_level", default="INFO")
    
    # Conservative performance parameters (stable defaults)
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Inference batch size (stable default: 64)")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of simulation workers (stable default: 16)")
    parser.add_argument("--prompt_queue_size", type=int, default=2048,
                       help="Prompt queue size (stable default: 2048)")
    parser.add_argument("--result_queue_size", type=int, default=1024,
                       help="Result queue size (stable default: 1024)")
    parser.add_argument("--queue_timeout", type=float, default=2.0,
                       help="Queue timeout in seconds (stable default: 2.0)")
    parser.add_argument("--batch_timeout", type=float, default=0.5,
                       help="Batch timeout in seconds (stable default: 0.5)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("🛡️  STABLE VERSION - Using float32 + eager attention for maximum stability")
    print("   This matches your working Mac Mini configuration")
    print()
    
    if args.single_match:
        print(f"Simulating single match {args.single_match} with stable configuration")
        # For single match, just call the series function with one match
        match_config = get_match_by_number(args.single_match)
        # Modify the simulate_match_series to handle single match
        all_matches_backup = get_all_matches
        def get_single_match():
            return [match_config]
        
        # Temporarily replace to simulate just one match
        import aus_wi_t20i_series_setup
        aus_wi_t20i_series_setup.get_all_matches = get_single_match
        
        try:
            simulate_match_series(
                args.model_path,
                args.simulations_per_match,
                f"{args.output_dir}_match_{args.single_match}",
                args.batch_size,
                args.num_workers,
                args.prompt_queue_size,
                32,  # batch_queue_size
                args.result_queue_size,
                args.queue_timeout,
                args.batch_timeout
            )
        finally:
            aus_wi_t20i_series_setup.get_all_matches = all_matches_backup
    else:
        # Simulate entire series
        simulate_match_series(
            args.model_path,
            args.simulations_per_match, 
            args.output_dir,
            args.batch_size,
            args.num_workers,
            args.prompt_queue_size,
            32,  # batch_queue_size
            args.result_queue_size,
            args.queue_timeout,
            args.batch_timeout
        )

"""
USAGE EXAMPLES FOR STABLE VERSION:

# Basic stable run (should work like Mac Mini):
python stable_series_wrapper.py --simulations_per_match 1000

# Conservative test run:
python stable_series_wrapper.py \
    --simulations_per_match 100 \
    --batch_size 32 \
    --num_workers 8

# Single match test:
python stable_series_wrapper.py --single_match 1 --simulations_per_match 100

# Larger stable run:
python stable_series_wrapper.py \
    --simulations_per_match 10000 \
    --batch_size 128 \
    --num_workers 32

NOTE: This version prioritizes stability over performance.
It uses float32 precision and eager attention like your working Mac Mini.
"""