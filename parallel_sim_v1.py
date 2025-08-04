#!/usr/bin/env python3
"""
Parallel Cricket Match Simulation System v2
Fixed version addressing critical issues

Changes from v1:
1. Fixed match ID collision bug with proper upfront distribution
2. Moved GPU inference to separate process (avoids CUDA context issues)
3. Reduced queue sizes to prevent memory bloat
4. Added comprehensive error handling and logging
5. Added atomic checkpoint writes
6. Added graceful shutdown with signal handlers
7. Fixed stats memory leak with running statistics
8. Added validation for incomplete simulations
9. Added worker result routing to prevent cross-contamination
10. Added queue monitoring and drain mechanisms

Remaining limitations:
- No retry mechanism for failed matches
- No dynamic load balancing if workers finish unevenly
- GPU process restarts load model each time
- No distributed multi-GPU support
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager, Event
import numpy as np
from dataclasses import dataclass, asdict
import time
import json
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import queue
import os
from pathlib import Path
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

# Import from your existing simulator
from simulator_v1 import (
    OUTCOMES, OUTCOME2TOK, TOK2OUTCOME, 
    Player, BallEvent, BatsmanStats, BowlerStats, MatchState,
    create_sample_teams
)

@dataclass
class SimulationConfig:
    """Configuration for parallel simulation"""
    num_matches: int = 10000
    batch_size: int = 128
    num_workers: int = 30
    prompt_queue_size: int = 1000  # Reduced as suggested
    batch_queue_size: int = 20     # Reduced
    result_queue_size: int = 500   # Reduced
    save_ball_by_ball: bool = False
    checkpoint_interval: int = 1000
    verbose: bool = True
    # New config options
    queue_timeout: float = 0.01
    batch_timeout: float = 0.01
    max_retries: int = 3

@dataclass
class PromptRequest:
    """Request for model inference"""
    match_id: int
    ball_id: int
    prompt: str

@dataclass 
class PredictionResult:
    """Result from model inference"""
    match_id: int
    ball_id: int
    outcome: str
    probabilities: Dict[str, float]

class MatchSimulator:
    """Single match simulator that generates prompts"""
    def __init__(self, match_id: int, team1_name: str, team2_name: str,
                 team1_players: List[Player], team2_players: List[Player],
                 venue: str = "Generic Stadium"):
        self.match_id = match_id
        self.match_state = MatchState(team1_name, team2_name, team1_players, team2_players, venue)
        self.ball_count = 0
        self.logger = logging.getLogger(f"Match-{match_id}")

        self.match_state.initialize_innings() 
        
    def get_next_prompt(self) -> Optional[PromptRequest]:
        """Get next prompt if match not complete"""
        try:
            # Check if match is complete
            if self.match_state.innings == 2:
                target = self.match_state.innings_scores.get(self.match_state.team1_name, {}).get("runs", 0) + 1
                if self.match_state.is_innings_complete() or self.match_state.score >= target:
                    return None
            elif self.match_state.innings == 1 and self.match_state.is_innings_complete():
                self.match_state.switch_innings()
                
            # Generate prompt
            prompt = self.match_state.generate_prompt()
            
            # Check if prompt is empty (all out case)
            if not prompt:
                return None
                
            request = PromptRequest(
                match_id=self.match_id,
                ball_id=self.ball_count,
                prompt=prompt
            )
            self.ball_count += 1
            return request
        except Exception as e:
            self.logger.error(f"Error generating prompt: {e}")
            return None
    
    def process_prediction(self, result: PredictionResult):
        """Process prediction result"""
        try:
            self.match_state.process_ball(result.outcome)
        except Exception as e:
            self.logger.error(f"Error processing prediction: {e}")
            raise
        
    def is_complete(self) -> bool:
        """Check if match simulation is complete"""
        if self.match_state.innings == 1:
            return False
        target = self.match_state.innings_scores.get(self.match_state.team1_name, {}).get("runs", 0) + 1
        return self.match_state.is_innings_complete() or self.match_state.score >= target
    
    def get_result(self) -> Dict:
        """Get match result"""
        # Store second innings score
        self.match_state.innings_scores[self.match_state.batting_team] = {
            "runs": self.match_state.score, 
            "wickets": self.match_state.wickets, 
            "overs": self.match_state.overs
        }
        
        # Determine winner
        team1_score = self.match_state.innings_scores.get(self.match_state.team1_name, {}).get("runs", 0)
        team2_score = self.match_state.innings_scores.get(self.match_state.team2_name, {}).get("runs", 0)
        
        if team2_score > team1_score:
            winner = self.match_state.team2_name
            margin = f"by {10 - self.match_state.innings_scores[self.match_state.team2_name]['wickets']} wickets"
        else:
            winner = self.match_state.team1_name
            margin = f"by {team1_score - team2_score} runs"
        
        # Convert dataclass objects to dicts for JSON serialization
        batting_stats_dict = {}
        for team, players in self.match_state.batting_stats.items():
            batting_stats_dict[team] = {}
            for player_name, stats in players.items():
                batting_stats_dict[team][player_name] = {
                    "runs": stats.runs,
                    "balls": stats.balls,
                    "fours": stats.fours,
                    "sixes": stats.sixes,
                    "strike_rate": stats.strike_rate
                }
        
        bowling_stats_dict = {}
        for team, players in self.match_state.bowling_stats.items():
            bowling_stats_dict[team] = {}
            for player_name, stats in players.items():
                bowling_stats_dict[team][player_name] = {
                    "overs": stats.overs,
                    "runs": stats.runs,
                    "wickets": stats.wickets,
                    "economy": stats.economy,
                    "balls": stats.balls
                }
        
        return {
            "match_id": self.match_id,
            "winner": winner,
            "margin": margin,
            "innings_scores": self.match_state.innings_scores,
            "batting_stats": batting_stats_dict,
            "bowling_stats": bowling_stats_dict,
            "total_balls": self.ball_count
        }

def simulation_worker(worker_id: int, config: SimulationConfig, 
                     prompt_queue: Queue, result_queue: Queue,
                     completed_queue: Queue, team1_players: List[Player], 
                     team2_players: List[Player], shutdown_event: Event):
    """Worker process that manages match simulations"""
    logger = logging.getLogger(f"Worker-{worker_id}")
    logger.info("Starting")
    
    # Fix: Proper match ID distribution
    matches_per_worker = config.num_matches // config.num_workers
    remainder = config.num_matches % config.num_workers
    
    # Distribute remainder evenly
    if worker_id < remainder:
        start_id = worker_id * (matches_per_worker + 1)
        end_id = start_id + matches_per_worker + 1
    else:
        start_id = worker_id * matches_per_worker + remainder
        end_id = start_id + matches_per_worker
    
    logger.info(f"Assigned match IDs: {start_id} to {end_id-1} ({end_id - start_id} matches)")
    
    # Track active matches and pending results
    active_matches = {}
    pending_results = defaultdict(list)  # match_id -> [results]
    matches_completed = 0
    last_log_time = time.time()
    
    # Create all assigned matches upfront
    for match_id in range(start_id, end_id):
        try:
            match = MatchSimulator(match_id, "India", "Sri Lanka", team1_players, team2_players)
            active_matches[match_id] = match
        except Exception as e:
            logger.error(f"Failed to create match {match_id}: {e}")
    
    logger.info(f"Created {len(active_matches)} matches")
    
    try:
        while (active_matches or pending_results) and not shutdown_event.is_set():
            # Generate prompts from active matches
            prompts_generated = 0
            for match_id, match in list(active_matches.items()):
                # Process any pending results first
                if match_id in pending_results:
                    for result in pending_results[match_id]:
                        match.process_prediction(result)
                    del pending_results[match_id]
                
                # Generate next prompt
                prompt_req = match.get_next_prompt()
                if prompt_req:
                    try:
                        prompt_queue.put(prompt_req, timeout=config.queue_timeout)
                        prompts_generated += 1
                    except queue.Full:
                        pass  # Queue full, try again later
                        
                # Check if match is complete
                if match.is_complete():
                    try:
                        result = match.get_result()
                        completed_queue.put(result, timeout=config.queue_timeout)
                        del active_matches[match_id]
                        matches_completed += 1
                        
                        # Log individual match completion
                        if matches_completed <= 10 or matches_completed % 50 == 0:
                            logger.info(f"Match {match_id} complete: {result['winner']} won {result['margin']}")
                            
                    except Exception as e:
                        logger.error(f"Failed to get result for match {match_id}: {e}")
            
            # Process incoming results
            results_processed = 0
            for _ in range(100):  # Process up to 100 results per iteration
                try:
                    result = result_queue.get(timeout=0.001)
                    
                    # Validate this result belongs to this worker
                    if not (start_id <= result.match_id < end_id):
                        # Put it back for the correct worker
                        result_queue.put(result)
                        continue
                    
                    if result.match_id in active_matches:
                        match = active_matches[result.match_id]
                        match.process_prediction(result)
                        results_processed += 1
                    else:
                        # Store for later if match will be created by this worker
                        pending_results[result.match_id].append(result)
                        
                        # Limit pending results to prevent memory growth
                        if len(pending_results[result.match_id]) > 100:
                            logger.warning(f"Too many pending results for match {result.match_id}")
                            pending_results[result.match_id] = pending_results[result.match_id][-50:]
                            
                except queue.Empty:
                    break
            
            # Periodic status log
            if time.time() - last_log_time > 10:  # Log every 10 seconds
                logger.info(f"Status: {matches_completed} completed, {len(active_matches)} active, "
                          f"{prompts_generated} prompts sent, {results_processed} results processed")
                last_log_time = time.time()
                    
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
    finally:
        logger.info(f"Completed {matches_completed} matches")

def gpu_inference_process(model_path: str, config: SimulationConfig,
                         prompt_queue: Queue, result_queue: Queue, 
                         shutdown_event: Event):
    """Separate process for GPU inference"""
    logger = logging.getLogger("GPU-Inference")
    logger.info("Starting GPU inference process")
    
    try:
        # Load model in this process to avoid context issues
        model, tokenizer, device = load_model_for_inference(model_path)
        
        batch_prompts = []
        batch_requests = []
        total_predictions = 0
        
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
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        inputs = tokenizer(batch_prompts, return_tensors="pt", 
                                         max_length=96, truncation=True, padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = model(**inputs)
                        logits = outputs.logits[:, -1, :]
                        
                        # Get outcome token IDs
                        outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
                        
                        # Process each prediction
                        for i, request in enumerate(batch_requests):
                            outcome_logits = logits[i, outcome_token_ids]
                            log_probs = torch.log_softmax(outcome_logits, dim=-1)
                            outcome_probs = torch.exp(log_probs).cpu().numpy()
                            
                            # Ensure valid probabilities
                            outcome_probs = np.clip(outcome_probs, 1e-8, 1.0)
                            outcome_probs = outcome_probs / outcome_probs.sum()
                            
                            # Sample outcome
                            outcome_idx = np.random.choice(len(OUTCOMES), p=outcome_probs)
                            
                            # Create result
                            result = PredictionResult(
                                match_id=request.match_id,
                                ball_id=request.ball_id,
                                outcome=OUTCOMES[outcome_idx],
                                probabilities={o: p for o, p in zip(OUTCOMES, outcome_probs)}
                            )
                            
                            result_queue.put(result)
                            total_predictions += 1
                    
                    # Clear batch
                    batch_prompts = []
                    batch_requests = []
                    
                    if total_predictions % 1000 == 0:
                        logger.info(f"Processed {total_predictions} predictions")
                        
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    # Clear failed batch
                    batch_prompts = []
                    batch_requests = []
                    
    except Exception as e:
        logger.error(f"GPU process failed: {e}", exc_info=True)
    finally:
        logger.info(f"Completed. Total predictions: {total_predictions}")

def result_aggregator(config: SimulationConfig, completed_queue: Queue, 
                     output_path: Path, shutdown_event: Event):
    """Aggregate results from completed matches"""
    logger = logging.getLogger("Aggregator")
    logger.info("Starting result aggregator")
    
    results = []
    # Fix: Use proper data structures
    stats = {
        "winners": defaultdict(int),
        "team_scores": defaultdict(list),
        "team_wickets": defaultdict(list),
        "score_summary": defaultdict(lambda: {"sum": 0, "count": 0, "sum_sq": 0}),
        "completed_matches": set()
    }
    
    try:
        matches_seen = 0
        last_progress_time = time.time()
        
        while matches_seen < config.num_matches:
            # Check for timeout
            if shutdown_event.is_set() and completed_queue.empty():
                if time.time() - last_progress_time > 30:  # 30 second timeout
                    logger.warning(f"No progress for 30s, stopping. Got {matches_seen}/{config.num_matches}")
                    break
            
            try:
                result = completed_queue.get(timeout=1.0)
                match_id = result["match_id"]
                
                # Validate not duplicate
                if match_id in stats["completed_matches"]:
                    logger.warning(f"Duplicate result for match {match_id}")
                    continue
                    
                stats["completed_matches"].add(match_id)
                results.append(result)
                matches_seen += 1
                last_progress_time = time.time()
                
                # Update statistics
                stats["winners"][result["winner"]] += 1
                
                # Team totals - store for checkpoint but also compute running stats
                for team, score in result["innings_scores"].items():
                    runs = score["runs"]
                    wickets = score["wickets"]
                    
                    # Store recent values for checkpoint
                    stats["team_scores"][team].append(runs)
                    stats["team_wickets"][team].append(wickets)
                    
                    # Keep only recent values to limit memory
                    if len(stats["team_scores"][team]) > 1000:
                        stats["team_scores"][team].pop(0)
                    if len(stats["team_wickets"][team]) > 1000:
                        stats["team_wickets"][team].pop(0)
                    
                    # Running statistics (memory efficient)
                    summary = stats["score_summary"][team]
                    summary["count"] += 1
                    summary["sum"] += runs
                    summary["sum_sq"] += runs * runs
                
                # Progress update
                if matches_seen % 100 == 0:
                    logger.info(f"Aggregated {matches_seen}/{config.num_matches} matches")
                    
                # Checkpoint with atomic write
                if matches_seen % config.checkpoint_interval == 0:
                    save_checkpoint_atomic(results, stats, output_path, matches_seen)
                    
            except queue.Empty:
                pass
                
    except Exception as e:
        logger.error(f"Aggregator failed: {e}", exc_info=True)
    finally:
        # Final save
        save_checkpoint_atomic(results, stats, output_path, matches_seen)
        logger.info(f"Completed. Total matches: {matches_seen}")
        
        # Validate completion
        if matches_seen < config.num_matches:
            logger.warning(f"Incomplete simulation: {matches_seen}/{config.num_matches}")
            missing = set(range(config.num_matches)) - stats["completed_matches"]
            if len(missing) < 20:
                logger.warning(f"Missing match IDs: {sorted(missing)}")
            else:
                logger.warning(f"Missing {len(missing)} matches")
        
        print_summary_stats(stats, matches_seen)

def save_checkpoint_atomic(results: List[Dict], stats: Dict, output_path: Path, 
                          checkpoint_num: int):
    """Save checkpoint with atomic write"""
    checkpoint_dir = output_path / f"checkpoint_{checkpoint_num}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary files first
    temp_results = checkpoint_dir / "results.json.tmp"
    temp_stats = checkpoint_dir / "statistics.json.tmp"
    
    try:
        # Save results
        with open(temp_results, "w") as f:
            json.dump(results[-1000:], f)
        
        # Save statistics with running averages
        stats_dict = {
            "total_matches": checkpoint_num,
            "winners": dict(stats["winners"]),
            "avg_scores": {},
            "score_std": {}
        }
        
        # Calculate statistics from running sums
        for team, summary in stats["score_summary"].items():
            if summary["count"] > 0:
                mean = summary["sum"] / summary["count"]
                variance = (summary["sum_sq"] / summary["count"]) - (mean * mean)
                std = np.sqrt(max(0, variance))  # Avoid negative variance due to float precision
                stats_dict["avg_scores"][team] = mean
                stats_dict["score_std"][team] = std
        
        with open(temp_stats, "w") as f:
            json.dump(stats_dict, f, indent=2)
        
        # Atomic rename
        temp_results.rename(checkpoint_dir / "results.json")
        temp_stats.rename(checkpoint_dir / "statistics.json")
        
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        # Clean up temp files
        for temp_file in [temp_results, temp_stats]:
            if temp_file.exists():
                temp_file.unlink()

def print_summary_stats(stats: Dict, total_matches: int):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal Matches Simulated: {total_matches}")
    
    if stats["winners"]:
        print("\nWin Distribution:")
        for team, wins in stats["winners"].items():
            win_pct = (wins / total_matches) * 100 if total_matches > 0 else 0
            print(f"  {team}: {wins} wins ({win_pct:.1f}%)")
    
    if stats["score_summary"]:
        print("\nAverage Scores:")
        for team, summary in stats["score_summary"].items():
            if summary["count"] > 0:
                mean = summary["sum"] / summary["count"]
                variance = (summary["sum_sq"] / summary["count"]) - (mean * mean)
                std = np.sqrt(max(0, variance))
                print(f"  {team}: {mean:.1f} ± {std:.1f}")
    
    # Completion status
    if "completed_matches" in stats:
        completion_rate = len(stats["completed_matches"]) / total_matches * 100 if total_matches > 0 else 0
        print(f"\nCompletion Rate: {completion_rate:.1f}%")

def load_model_for_inference(model_path: str):
    """Load model optimized for inference"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    base_model_name = "Qwen/Qwen1.5-1.8B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Add special tokens
    special_tokens = list(OUTCOME2TOK.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=False,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # Resize embeddings
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)
    model.eval()
    
    # Compile for optimization
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully")
        except:
            print("Compilation failed, using eager mode")
    
    return model, tokenizer, device

def signal_handler(signum, frame, shutdown_event):
    """Handle shutdown signals gracefully"""
    logging.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()

def run_parallel_simulation(model_path: str, config: SimulationConfig, 
                          output_path: Path):
    """Main function to run parallel simulation"""
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
    
    # from real_teams_setup import create_wi_vs_pak_teams
    # team1_players, team2_players = create_wi_vs_pak_teams()
    team1_players, team2_players = create_sample_teams()
    
    # Create manager and queues
    manager = Manager()
    prompt_queue = manager.Queue(config.prompt_queue_size)
    result_queue = manager.Queue(config.result_queue_size)
    completed_queue = manager.Queue(config.result_queue_size)
    
    # Create shutdown event
    shutdown_event = manager.Event()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, shutdown_event))
    signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, shutdown_event))
    
    # Start time
    start_time = time.time()
    
    processes = []
    
    try:
        # Start simulation workers
        for i in range(config.num_workers):
            p = Process(target=simulation_worker, args=(
                i, config, prompt_queue, result_queue, completed_queue,
                team1_players, team2_players, shutdown_event
            ))
            p.start()
            processes.append(p)
        
        # Start GPU inference process (not thread!)
        gpu_process = Process(target=gpu_inference_process, args=(
            model_path, config, prompt_queue, result_queue, shutdown_event
        ))
        gpu_process.start()
        processes.append(gpu_process)
        
        # Start result aggregator process
        agg_process = Process(target=result_aggregator, args=(
            config, completed_queue, output_path, shutdown_event
        ))
        agg_process.start()
        processes.append(agg_process)
        
        # Monitor progress
        workers_alive = config.num_workers
        while workers_alive > 0:
            time.sleep(5)
            workers_alive = sum(1 for p in processes[:config.num_workers] if p.is_alive())
            
            if workers_alive == 0:
                logging.info("All workers completed")
                break
                
            # Log queue depths
            try:
                logging.info(f"Queue depths - Prompts: ~{prompt_queue.qsize()}, "
                           f"Results: ~{result_queue.qsize()}, "
                           f"Completed: ~{completed_queue.qsize()}")
            except:
                pass  # qsize not available on all platforms
        
        # Workers done, let GPU drain the queue
        logging.info("Waiting for GPU to process remaining prompts...")
        time.sleep(5)  # Give time for queue to drain
        
        # Signal shutdown
        shutdown_event.set()
        
        # Wait for all processes with timeout
        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                logging.warning(f"Process {p.name} did not exit gracefully")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        
        logging.info("All processes completed")
        
    except KeyboardInterrupt:
        logging.info("Interrupted, shutting down...")
        shutdown_event.set()
        
    finally:
        # Ensure all processes are terminated
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
        
        # Print timing
        total_time = time.time() - start_time
        print(f"\nTotal simulation time: {total_time:.1f} seconds")
        
        # Validate output exists
        checkpoints = list(output_path.glob("checkpoint_*"))
        if checkpoints:
            latest = sorted(checkpoints)[-1]
            with open(latest / "statistics.json") as f:
                final_stats = json.load(f)
            completed = final_stats.get("total_matches", 0)
            print(f"Matches completed: {completed}/{config.num_matches}")
            if completed > 0:
                print(f"Matches per second: {completed / total_time:.1f}")
        else:
            print("WARNING: No output generated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="qwen-cricket-peft-2")
    parser.add_argument("--num_matches", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=30)
    parser.add_argument("--output_dir", default="simulation_results")
    parser.add_argument("--log_level", default="INFO")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    config = SimulationConfig(
        num_matches=args.num_matches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_ball_by_ball=False
    )
    
    output_path = Path(args.output_dir) / f"sim_{args.num_matches}_{int(time.time())}"
    
    print(f"Starting parallel simulation:")
    print(f"  Matches: {config.num_matches}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Output: {output_path}")
    
    run_parallel_simulation(args.model_path, config, output_path)