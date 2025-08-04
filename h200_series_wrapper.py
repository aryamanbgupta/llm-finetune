#!/usr/bin/env python3
"""
Series Simulation Wrapper - H200 Optimized (FIXED Pickle Issue)
Uses bfloat16 + flash attention + torch.compile with robust NaN handling
FIXED: Module-level functions to avoid multiprocessing pickle errors
"""

import time
import sys
from pathlib import Path
import argparse
import logging
import os

# Import required modules
from aus_wi_t20i_series_setup import get_all_matches, get_match_by_number

# FIXED: Define patched functions at MODULE LEVEL to avoid pickle issues

def load_model_for_inference_h200_fixed(model_path: str):
    """H200 optimized model loading with robust error handling"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from parallel_sim_v1 import OUTCOME2TOK
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} with H200 optimizations...")
    
    # Set environment variables for better torch.compile behavior
    os.environ['TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS'] = '1'
    os.environ['TORCHDYNAMO_VERBOSE'] = '0'
    
    base_model_name = "Qwen/Qwen1.5-1.8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Add special tokens
    special_tokens = list(OUTCOME2TOK.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    # H200 OPTIMIZED: Load with bfloat16 and flash attention
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,  # H200 optimized
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="flash_attention_2"
    )
    
    # Resize embeddings
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # CRITICAL: Ensure precision consistency across all parameters
    print("Ensuring precision consistency...")
    inconsistent_params = 0
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            inconsistent_params += 1
            param.data = param.data.to(torch.bfloat16)
    
    if inconsistent_params > 0:
        print(f"✅ Converted {inconsistent_params} parameters to bfloat16 for consistency")
    
    model.eval()
    
    # H200 OPTIMIZED: torch.compile with robust settings
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            print("Compiling model for H200 with robust settings...")
            model = torch.compile(
                model, 
                mode='reduce-overhead',  # Balanced optimization
                dynamic=True,           # Handle varying batch sizes
                fullgraph=False        # Allow graph breaks for stability
            )
            print("✅ Model compiled successfully for H200")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}, using eager mode")
    
    # Validate special tokens are properly loaded
    test_tokens = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
    if any(tid == tokenizer.unk_token_id for tid in test_tokens):
        raise ValueError("Some outcome tokens not found in tokenizer after loading!")
    print(f"✅ All {len(test_tokens)} special tokens validated")
    
    return model, tokenizer, device

def gpu_inference_process_h200_fixed(model_path: str, config, prompt_queue, result_queue, shutdown_event):
    """H200 optimized GPU inference with comprehensive error handling"""
    import torch
    import numpy as np
    import queue
    import time
    import logging
    from parallel_sim_v1 import OUTCOMES, PredictionResult, OUTCOME2TOK
    
    logger = logging.getLogger("GPU-Inference")
    logger.info("Starting H200 optimized GPU inference process")
    
    try:
        # Load model in this process
        model, tokenizer, device = load_model_for_inference_h200_fixed(model_path)
        
        batch_prompts = []
        batch_requests = []
        total_predictions = 0
        nan_count = 0
        
        # Get outcome token IDs once
        outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
        outcome_token_ids = torch.tensor(outcome_token_ids, device=device)
        
        while not shutdown_event.is_set() or not prompt_queue.empty():
            # Collect batch
            deadline = time.time() + config.batch_timeout
            
            while len(batch_prompts) < config.batch_size and time.time() < deadline:
                try:
                    request = prompt_queue.get(timeout=0.001)
                    batch_prompts.append(request.prompt)
                    batch_requests.append(request)
                except queue.Empty:
                    if shutdown_event.is_set():
                        break
            
            # Process batch
            if batch_prompts:
                try:
                    # FIXED: Updated autocast syntax for H200
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        inputs = tokenizer(batch_prompts, return_tensors="pt", 
                                         max_length=96, truncation=True, padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = model(**inputs)
                        logits = outputs.logits[:, -1, :]
                        
                        # Extract outcome logits for all samples at once
                        outcome_logits = logits[:, outcome_token_ids]  # [batch_size, num_outcomes]
                        
                        # Process each prediction with robust error handling
                        for i, request in enumerate(batch_requests):
                            sample_logits = outcome_logits[i]
                            
                            # COMPREHENSIVE: Check for invalid values
                            has_nan = torch.isnan(sample_logits).any()
                            has_inf = torch.isinf(sample_logits).any()
                            
                            if has_nan or has_inf:
                                nan_count += 1
                                if nan_count <= 10:  # Log first few occurrences
                                    logger.warning(f"Invalid logits for match {request.match_id} ball {request.ball_id}: "
                                                 f"NaN={has_nan}, Inf={has_inf}")
                                # Use uniform distribution
                                outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
                            else:
                                # Compute probabilities with numerical stability
                                try:
                                    # Use log-softmax for numerical stability
                                    log_probs = torch.log_softmax(sample_logits, dim=-1)
                                    probs_tensor = torch.exp(log_probs)
                                    outcome_probs = probs_tensor.cpu().numpy()
                                    
                                    # Validate probabilities
                                    if np.isnan(outcome_probs).any() or np.isinf(outcome_probs).any():
                                        if nan_count <= 10:
                                            logger.warning(f"NaN/Inf in probabilities for match {request.match_id}")
                                        nan_count += 1
                                        outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
                                    else:
                                        # Ensure probabilities are valid
                                        outcome_probs = np.clip(outcome_probs, 1e-8, 1.0)
                                        prob_sum = outcome_probs.sum()
                                        
                                        if prob_sum == 0 or np.isnan(prob_sum) or np.isinf(prob_sum):
                                            if nan_count <= 10:
                                                logger.warning(f"Invalid probability sum: {prob_sum}")
                                            nan_count += 1
                                            outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
                                        else:
                                            outcome_probs = outcome_probs / prob_sum
                                            
                                except Exception as e:
                                    if nan_count <= 10:
                                        logger.warning(f"Probability computation failed: {e}")
                                    nan_count += 1
                                    outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
                            
                            # Final validation before sampling
                            if not np.allclose(outcome_probs.sum(), 1.0, atol=1e-6):
                                outcome_probs = outcome_probs / outcome_probs.sum()
                            
                            # Sample outcome
                            try:
                                outcome_idx = np.random.choice(len(OUTCOMES), p=outcome_probs)
                                predicted_outcome = OUTCOMES[outcome_idx]
                            except Exception as e:
                                logger.error(f"Sampling failed: {e}, using fallback")
                                predicted_outcome = np.random.choice(OUTCOMES)
                                outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
                            
                            # Create result
                            result = PredictionResult(
                                match_id=request.match_id,
                                ball_id=request.ball_id,
                                outcome=predicted_outcome,
                                probabilities={o: float(p) for o, p in zip(OUTCOMES, outcome_probs)}
                            )
                            
                            result_queue.put(result)
                            total_predictions += 1
                    
                    # Clear batch
                    batch_prompts = []
                    batch_requests = []
                    
                    # Periodic logging
                    if total_predictions % 1000 == 0:
                        if nan_count > 0:
                            logger.info(f"Processed {total_predictions} predictions ({nan_count} with NaN/Inf)")
                        else:
                            logger.info(f"Processed {total_predictions} predictions")
                        
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    # Clear failed batch and continue
                    batch_prompts = []
                    batch_requests = []
                    
    except Exception as e:
        logger.error(f"GPU process failed: {e}", exc_info=True)
    finally:
        if nan_count > 0:
            logger.warning(f"Total NaN/Inf occurrences: {nan_count}/{total_predictions} ({nan_count/max(total_predictions,1)*100:.1f}%)")
        logger.info(f"H200 GPU inference completed. Total predictions: {total_predictions}")

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
    """Simulate all 5 matches with H200 optimization (FIXED)"""
    
    print("="*80)
    print("AUSTRALIA vs WEST INDIES T20I SERIES SIMULATION - H200 OPTIMIZED (FIXED)")
    print("="*80)
    print(f"Simulations per match: {simulations_per_match:,}")
    print(f"Total simulations: {simulations_per_match * 5:,}")
    print(f"Output directory: {base_output_dir}")
    print()
    
    # FIXED: Patch at module level to avoid pickle issues
    import parallel_sim_v1
    
    # Store original functions
    original_load_model = getattr(parallel_sim_v1, 'load_model_for_inference', None)
    original_gpu_process = getattr(parallel_sim_v1, 'gpu_inference_process', None)
    
    # FIXED: Monkey patch with module-level functions (picklable)
    parallel_sim_v1.load_model_for_inference = load_model_for_inference_h200_fixed
    parallel_sim_v1.gpu_inference_process = gpu_inference_process_h200_fixed
    print("✅ Applied H200 optimizations to parallel_sim_v1")
    
    try:
        # Import after patching
        from parallel_sim_v1 import SimulationConfig, run_parallel_simulation
        
        # Get all match configurations
        all_matches = get_all_matches()
        base_path = Path(base_output_dir)
        base_path.mkdir(exist_ok=True)
        
        # H200 optimized configuration
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
        
        print(f"H200 Optimized Configuration (FIXED):")
        print(f"  Batch size: {config.batch_size} (H200 optimized)")
        print(f"  Workers: {config.num_workers} (H200 optimized)")
        print(f"  Prompt queue: {config.prompt_queue_size:,}")
        print(f"  Result queue: {config.result_queue_size:,}")
        print(f"  Checkpoint interval: {config.checkpoint_interval}")
        print(f"  Precision: bfloat16 + flash attention + torch.compile")
        print(f"  Error handling: Comprehensive NaN/Inf detection and fallbacks")
        print(f"  FIXED: Module-level functions (no pickle errors)")
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
        
    finally:
        # RESTORE: Put back original functions
        if original_load_model:
            parallel_sim_v1.load_model_for_inference = original_load_model
        if original_gpu_process:
            parallel_sim_v1.gpu_inference_process = original_gpu_process
        print("✅ Restored original parallel_sim_v1 functions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate AUS vs WI T20I series - H200 Optimized (FIXED)")
    
    # Basic options
    parser.add_argument("--model_path", default="models/checkpoint-24000", 
                       help="Path to fine-tuned model")
    parser.add_argument("--simulations_per_match", type=int, default=100,  # Default to small for testing
                       help="Number of simulations per match")
    parser.add_argument("--output_dir", default="series_simulation_results_h200_fixed",
                       help="Base output directory")
    parser.add_argument("--single_match", type=int, choices=[1,2,3,4,5],
                       help="Simulate only a specific match (1-5)")
    parser.add_argument("--log_level", default="INFO")
    
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
    
    print("🚀 H200 OPTIMIZED VERSION (FIXED) - No more pickle errors!")
    print("   Using bfloat16 + flash attention + torch.compile + module-level functions")
    print()
    
    if args.single_match:
        print(f"Simulating single match {args.single_match} with H200 optimization")
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
                args.batch_queue_size,
                args.result_queue_size,
                args.queue_timeout,
                args.batch_timeout,
                args.checkpoint_interval
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
            args.batch_queue_size,
            args.result_queue_size,
            args.queue_timeout,
            args.batch_timeout,
            args.checkpoint_interval
        )

"""
USAGE EXAMPLES FOR FIXED H200 VERSION:

# Quick test (no pickle errors):
python fixed_h200_wrapper.py --simulations_per_match 100

# Single match test:
python fixed_h200_wrapper.py --single_match 1 --simulations_per_match 100

# H200 production run:
python fixed_h200_wrapper.py --simulations_per_match 50000

FIXED ISSUES:
✅ Moved nested functions to module level (no pickle errors)
✅ Proper function restoration in finally block
✅ Reduced default simulations for testing (10000 -> 100)
✅ Same shutdown improvements as stable version
✅ All H200 optimizations preserved
"""