#!/usr/bin/env python3
"""
Series Simulation Wrapper - H200 Optimized Defaults
Runs parallel simulations for all 5 AUS vs WI T20I matches
10 simulations per match, stored in separate folders
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

def simulate_match_series(model_path: str, simulations_per_match: int = 10000, 
                         base_output_dir: str = "series_simulation_results",
                         batch_size: int = 512,
                         num_workers: int = 64,
                         prompt_queue_size: int = 8192,
                         batch_queue_size: int = 128,
                         result_queue_size: int = 4096,
                         queue_timeout: float = 0.5,
                         batch_timeout: float = 0.1,
                         checkpoint_interval: int = None):
    """
    Simulate all 5 matches from AUS vs WI series
    
    Args:
        model_path: Path to your fine-tuned model
        simulations_per_match: Number of simulations per match
        base_output_dir: Base directory for all results
        All other args: H200-optimized configuration parameters
    """
    
    print("="*80)
    print("AUSTRALIA vs WEST INDIES T20I SERIES SIMULATION")
    print("="*80)
    print(f"Simulations per match: {simulations_per_match:,}")
    print(f"Total simulations: {simulations_per_match * 5:,}")
    print(f"Output directory: {base_output_dir}")
    print()
    
    # Get all match configurations
    all_matches = get_all_matches()
    base_path = Path(base_output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Configuration for each simulation (H200 optimized defaults)
    if checkpoint_interval is None:
        checkpoint_interval = max(100, simulations_per_match // 20)
    
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
    
    print(f"H200 Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Prompt queue: {config.prompt_queue_size:,}")
    print(f"  Result queue: {config.result_queue_size:,}")
    print(f"  Checkpoint interval: {config.checkpoint_interval}")
    print(f"  Queue timeout: {config.queue_timeout}s")
    print(f"  Batch timeout: {config.batch_timeout}s")
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
        original_create_teams = parallel_sim_v1.create_sample_teams
        
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
    
    # List all created directories
    print(f"\nOutput structure:")
    for match_dir in sorted(base_path.glob("match_*")):
        sim_dirs = list(match_dir.glob("sim_*"))
        print(f"  {match_dir.name}/")
        for sim_dir in sim_dirs:
            checkpoints = len(list(sim_dir.glob("checkpoint_*")))
            print(f"    {sim_dir.name}/ ({checkpoints} checkpoints)")

def simulate_single_match(match_number: int, model_path: str, 
                         simulations: int = 10000, output_dir: str = "single_match_sim",
                         batch_size: int = 512,
                         num_workers: int = 64,
                         prompt_queue_size: int = 8192,
                         batch_queue_size: int = 128,
                         result_queue_size: int = 4096,
                         queue_timeout: float = 0.5,
                         batch_timeout: float = 0.1,
                         checkpoint_interval: int = None):
    """
    Simulate a single specific match from the series
    
    Args:
        match_number: Which match to simulate (1-5)
        model_path: Path to model
        simulations: Number of simulations 
        output_dir: Output directory
        All other args: H200-optimized configuration parameters
    """
    
    match_config = get_match_by_number(match_number)
    
    print(f"Simulating {match_config['match_name']}")
    print(f"Teams: {match_config['team1_name']} vs {match_config['team2_name']}")
    print(f"Venue: {match_config['venue']}, {match_config['location']}")
    
    # Use same monkey patch approach
    import parallel_sim_v1
    original_create_teams = parallel_sim_v1.create_sample_teams
    match_teams_func = create_match_teams_function(match_config)
    parallel_sim_v1.create_sample_teams = match_teams_func
    
    try:
        if checkpoint_interval is None:
            checkpoint_interval = max(100, simulations // 20)
            
        config = SimulationConfig(
            num_matches=simulations,
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
        
        output_path = Path(output_dir) / f"match_{match_number}_{int(time.time())}"
        run_parallel_simulation(model_path, config, output_path)
        
    finally:
        parallel_sim_v1.create_sample_teams = original_create_teams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate AUS vs WI T20I series - H200 Optimized")
    
    # Basic options
    parser.add_argument("--model_path", default="models/checkpoint-24000", 
                       help="Path to fine-tuned model")
    parser.add_argument("--simulations_per_match", type=int, default=10000,
                       help="Number of simulations per match")
    parser.add_argument("--output_dir", default="series_simulation_results",
                       help="Base output directory")
    parser.add_argument("--single_match", type=int, choices=[1,2,3,4,5],
                       help="Simulate only a specific match (1-5)")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # H200 Performance parameters (with H200-optimized defaults)
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Inference batch size (H200 default: 512)")
    parser.add_argument("--num_workers", type=int, default=64,
                       help="Number of simulation workers (H200 default: 64)")
    parser.add_argument("--prompt_queue_size", type=int, default=8192,
                       help="Prompt queue size (H200 default: 8192)")
    parser.add_argument("--batch_queue_size", type=int, default=128,
                       help="Batch queue size (H200 default: 128)")
    parser.add_argument("--result_queue_size", type=int, default=4096,
                       help="Result queue size (H200 default: 4096)")
    parser.add_argument("--queue_timeout", type=float, default=0.5,
                       help="Queue timeout in seconds (H200 default: 0.5)")
    parser.add_argument("--batch_timeout", type=float, default=0.1,
                       help="Batch timeout in seconds (H200 default: 0.1)")
    parser.add_argument("--checkpoint_interval", type=int, default=None,
                       help="Checkpoint every N matches (default: auto-calculated)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    if args.single_match:
        # Simulate single match
        simulate_single_match(
            args.single_match, 
            args.model_path,
            args.simulations_per_match,
            f"{args.output_dir}_match_{args.single_match}",
            args.batch_size,
            args.num_workers,
            args.prompt_queue_size,
            args.batch_queue_size,
            args.result_queue_size,
            args.queue_timeout,
            args.batch_timeout,
            args.checkpoint_interval
        )
    else:
        # Simulate entire series
        simulate_match_series(
            args.model_path,
            args.simulations_per_match, 
            args.output_dir,
            args.batch_size,
            args.num_workers,
            args.prompt_queue_size,
            args.batch_queue_size,
            args.result_queue_size,
            args.queue_timeout,
            args.batch_timeout,
            args.checkpoint_interval
        )

"""
USAGE EXAMPLES:

# H200 defaults - 50k simulations per match:
python series_simulation_wrapper.py --simulations_per_match 50000

# Custom batch size for maximum throughput:
python series_simulation_wrapper.py --batch_size 768 --simulations_per_match 100000

# Single match with custom workers:
python series_simulation_wrapper.py --single_match 3 --num_workers 96 --batch_size 1024

# Conservative settings for testing:
python series_simulation_wrapper.py --batch_size 128 --num_workers 16 --simulations_per_match 1000

# Maximum throughput configuration:
python series_simulation_wrapper.py \
    --batch_size 768 \
    --num_workers 96 \
    --prompt_queue_size 16384 \
    --result_queue_size 8192 \
    --queue_timeout 0.2 \
    --batch_timeout 0.05 \
    --simulations_per_match 100000
"""