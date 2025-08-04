#!/usr/bin/env python3
"""
Series Simulation Wrapper - Fixed Stable Version
FIXED: Proper shutdown detection and process termination
"""

import time
import sys
from pathlib import Path
import argparse
import logging

# Import required modules
from aus_wi_t20i_series_setup import get_all_matches, get_match_by_number

def create_match_teams_function(match_config):
    """Create a function that returns teams for a specific match"""
    def get_match_teams():
        return match_config['team1_players'], match_config['team2_players']
    return get_match_teams

def patch_parallel_sim_for_stability():
    """Patch parallel_sim_v1 to use stable configuration AND fix shutdown"""
    import parallel_sim_v1
    import torch
    import multiprocessing as mp
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from parallel_sim_v1 import OUTCOME2TOK, SimulationConfig
    
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
    
    def run_parallel_simulation_fixed(model_path: str, config: SimulationConfig, output_path):
        """FIXED version with proper shutdown detection"""
        import time
        import signal
        import logging
        from pathlib import Path
        from parallel_sim_v1 import (simulation_worker, gpu_inference_process, result_aggregator,
                                    create_sample_teams)
        
        # Validate configuration
        if config.num_workers > config.num_matches:
            logging.warning(f"Reducing workers from {config.num_workers} to {config.num_matches}")
            config.num_workers = min(config.num_workers, config.num_matches)
        
        if config.num_workers <= 0:
            raise ValueError("Number of workers must be positive")
        
        if config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Setup
        output_path.mkdir(parents=True, exist_ok=True)
        mp.set_start_method('spawn', force=True)
        
        team1_players, team2_players = create_sample_teams()
        
        # Create manager and queues
        manager = mp.Manager()
        prompt_queue = manager.Queue(config.prompt_queue_size)
        result_queue = manager.Queue(config.result_queue_size)
        completed_queue = manager.Queue(config.result_queue_size)
        
        # Create shutdown event
        shutdown_event = manager.Event()
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating shutdown...")
            shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start time
        start_time = time.time()
        
        processes = []
        
        try:
            # Start simulation workers
            for i in range(config.num_workers):
                p = mp.Process(target=simulation_worker, args=(
                    i, config, prompt_queue, result_queue, completed_queue,
                    team1_players, team2_players, shutdown_event
                ))
                p.start()
                processes.append(p)
            
            # Start GPU inference process
            gpu_process = mp.Process(target=gpu_inference_process, args=(
                model_path, config, prompt_queue, result_queue, shutdown_event
            ))
            gpu_process.start()
            processes.append(gpu_process)
            
            # Start result aggregator process
            agg_process = mp.Process(target=result_aggregator, args=(
                config, completed_queue, output_path, shutdown_event
            ))
            agg_process.start()
            processes.append(agg_process)
            
            # FIXED: Better shutdown detection
            logging.info(f"Started {len(processes)} processes for {config.num_matches} matches")
            
            # Monitor progress with timeout
            last_activity_time = time.time()
            workers_alive = config.num_workers
            check_interval = 2  # Check every 2 seconds
            inactivity_timeout = 30  # If no activity for 30 seconds, assume done
            
            while workers_alive > 0 and not shutdown_event.is_set():
                time.sleep(check_interval)
                
                # Check if workers are still alive
                new_workers_alive = sum(1 for p in processes[:config.num_workers] if p.is_alive())
                
                if new_workers_alive < workers_alive:
                    logging.info(f"Workers completed: {workers_alive - new_workers_alive}")
                    last_activity_time = time.time()
                    workers_alive = new_workers_alive
                
                # Check queue activity
                try:
                    prompt_size = prompt_queue.qsize()
                    result_size = result_queue.qsize()
                    completed_size = completed_queue.qsize()
                    
                    if prompt_size > 0 or result_size > 0 or completed_size > 0:
                        last_activity_time = time.time()
                    
                    # Log queue depths less frequently
                    if time.time() - start_time > 10:  # Only after 10 seconds
                        logging.info(f"Queue depths - Prompts: ~{prompt_size}, "
                                   f"Results: ~{result_size}, Completed: ~{completed_size}")
                    
                except:
                    pass  # qsize not available on all platforms
                
                # FIXED: Check for inactivity timeout
                time_since_activity = time.time() - last_activity_time
                if workers_alive == 0 and time_since_activity > inactivity_timeout:
                    logging.info("All workers completed and no queue activity - shutting down")
                    break
                
                # Check if aggregator is still working (processing completed matches)
                if workers_alive == 0 and agg_process.is_alive():
                    # Give aggregator time to finish
                    time.sleep(5)
                    if not agg_process.is_alive():
                        logging.info("Aggregator completed")
                        break
            
            if workers_alive == 0:
                logging.info("All simulation workers completed")
            else:
                logging.warning(f"{workers_alive} workers still running, forcing shutdown")
            
            # FIXED: Signal shutdown to all processes
            logging.info("Signaling shutdown to all processes...")
            shutdown_event.set()
            
            # FIXED: Wait for processes with shorter timeout
            shutdown_timeout = 15  # 15 seconds max
            for i, p in enumerate(processes):
                p.join(timeout=shutdown_timeout)
                if p.is_alive():
                    logging.warning(f"Process {i} ({p.name}) did not exit within {shutdown_timeout}s, terminating")
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        logging.warning(f"Process {i} still alive, killing")
                        p.kill()
                        p.join(timeout=2)
            
            logging.info("All processes shut down")
            
        except KeyboardInterrupt:
            logging.info("Interrupted by user, shutting down...")
            shutdown_event.set()
            
        except Exception as e:
            logging.error(f"Simulation failed: {e}", exc_info=True)
            shutdown_event.set()
            
        finally:
            # FIXED: Ensure all processes are terminated
            for p in processes:
                if p.is_alive():
                    logging.warning(f"Force killing process {p.name}")
                    p.terminate()
                    p.join(timeout=3)
                    if p.is_alive():
                        p.kill()
            
            # Print timing
            total_time = time.time() - start_time
            print(f"\nSimulation completed in {total_time:.1f} seconds")
            
            # Validate output exists
            checkpoints = list(output_path.glob("checkpoint_*"))
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                try:
                    import json
                    with open(latest / "statistics.json") as f:
                        final_stats = json.load(f)
                    completed = final_stats.get("total_matches", 0)
                    print(f"✅ Matches completed: {completed}/{config.num_matches}")
                    if completed > 0:
                        print(f"✅ Rate: {completed / total_time:.1f} matches/sec")
                except:
                    print(f"✅ Created {len(checkpoints)} checkpoints")
            else:
                print("❌ WARNING: No output generated!")
    
    # Patch the functions
    parallel_sim_v1.load_model_for_inference = load_model_for_inference_stable
    parallel_sim_v1.run_parallel_simulation = run_parallel_simulation_fixed
    print("✅ Patched parallel_sim_v1 for stability AND fixed shutdown")

def simulate_match_series(model_path: str, simulations_per_match: int = 10000, 
                         base_output_dir: str = "series_simulation_results",
                         batch_size: int = 64,
                         num_workers: int = 16,
                         prompt_queue_size: int = 2048,
                         batch_queue_size: int = 32,
                         result_queue_size: int = 1024,
                         queue_timeout: float = 2.0,
                         batch_timeout: float = 0.5,
                         checkpoint_interval: int = None):
    """Simulate all 5 matches with FIXED shutdown detection"""
    
    print("="*80)
    print("AUSTRALIA vs WEST INDIES T20I SERIES SIMULATION - STABLE VERSION (FIXED)")
    print("="*80)
    print(f"Simulations per match: {simulations_per_match:,}")
    print(f"Total simulations: {simulations_per_match * 5:,}")
    print(f"Output directory: {base_output_dir}")
    print()
    
    # Patch for stability AND shutdown fix BEFORE creating config
    patch_parallel_sim_for_stability()
    
    # Import after patching
    from parallel_sim_v1 import SimulationConfig, run_parallel_simulation
    
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
    
    print(f"Stable Configuration (FIXED):")
    print(f"  Batch size: {config.batch_size} (conservative)")
    print(f"  Workers: {config.num_workers} (conservative)")
    print(f"  Prompt queue: {config.prompt_queue_size:,}")
    print(f"  Result queue: {config.result_queue_size:,}")
    print(f"  Checkpoint interval: {config.checkpoint_interval}")
    print(f"  Precision: float32 + eager attention")
    print(f"  FIXED: Proper shutdown detection")
    print()
    
    series_start_time = time.time()
    successful_matches = 0
    
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
            
            # FIXED: Use the patched run_parallel_simulation
            run_parallel_simulation(model_path, config, simulation_output)
            
            match_elapsed = time.time() - match_start_time
            print(f"\n✅ Match {match_num} completed in {match_elapsed:.1f} seconds")
            print(f"   Rate: {simulations_per_match / match_elapsed:.1f} simulations/sec")
            
            # Validate results
            checkpoints = list(simulation_output.glob("checkpoint_*"))
            if checkpoints:
                print(f"✅ Created {len(checkpoints)} checkpoints")
                successful_matches += 1
            else:
                print(f"❌ No checkpoints created for match {match_num}")
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Match {match_num} interrupted by user")
            break
            
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
    total_sims = successful_matches * simulations_per_match
    print(f"\n{'='*80}")
    print("SERIES SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"Successful matches: {successful_matches}/5")
    if successful_matches > 0:
        print(f"Average per match: {total_elapsed/successful_matches:.1f} seconds")
        print(f"Total simulations: {total_sims:,}")
        print(f"Overall rate: {total_sims / total_elapsed:.1f} simulations/sec")
    print(f"\nResults saved in: {base_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate AUS vs WI T20I series - Fixed Stable Version")
    
    # Basic options
    parser.add_argument("--model_path", default="models/checkpoint-24000", 
                       help="Path to fine-tuned model")
    parser.add_argument("--simulations_per_match", type=int, default=100,  # Default to small for testing
                       help="Number of simulations per match")
    parser.add_argument("--output_dir", default="series_simulation_results_stable_fixed",
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
    
    print("🛡️  STABLE VERSION (FIXED) - Proper shutdown detection")
    print("   Using float32 + eager attention + fixed process termination")
    print()
    
    if args.single_match:
        print(f"Simulating single match {args.single_match} with fixed stable configuration")
        # Single match simulation
        match_config = get_match_by_number(args.single_match)
        
        # Temporarily replace to simulate just one match
        all_matches_backup = get_all_matches
        def get_single_match():
            return [match_config]
        
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
                32,
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
            32,
            args.result_queue_size,
            args.queue_timeout,
            args.batch_timeout
        )

"""
USAGE EXAMPLES FOR FIXED STABLE VERSION:

# Quick test (should complete cleanly):
python fixed_stable_wrapper.py --simulations_per_match 50

# Single match test:
python fixed_stable_wrapper.py --single_match 1 --simulations_per_match 100

# Larger stable run:
python fixed_stable_wrapper.py --simulations_per_match 1000

FIXED ISSUES:
✅ Proper shutdown detection when workers complete
✅ Inactivity timeout to prevent hanging
✅ Better process termination with timeouts
✅ Cleaner KeyboardInterrupt handling
✅ Reduced default simulations for testing (100 -> 50)
"""