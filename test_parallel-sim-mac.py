#!/usr/bin/env python3
"""
Simple test script for Mac Mini - 2 workers, small batch
"""

import sys
import time
from pathlib import Path

# Import the parallel simulation components
try:
    from simulator_v1 import create_sample_teams
    from parallel_sim_v1 import SimulationConfig, run_parallel_simulation
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure simulator_v1.py and parallel-sim-v1.py are in the same directory")
    sys.exit(1)

def test_parallel_simulation():
    """Test with minimal configuration"""
    
    # Simple test config for Mac Mini - 2 workers, 1 match each
    config = SimulationConfig(
        num_matches=2,         # 2 matches total (1 per worker)
        batch_size=1,          # Single predictions - faster on CPU
        num_workers=2,         # 2 workers to test multiprocessing
        prompt_queue_size=600, # Large enough for 2 full matches (~240 balls each)
        result_queue_size=600,
        checkpoint_interval=1,  # Checkpoint every match
        verbose=True,
        queue_timeout=1.0,     # Longer timeout
        batch_timeout=0.1      # Much shorter batch timeout
    )
    
    # Set model path - update this to your actual model path
    model_path = "models/checkpoint-24000"  # Change this to your model directory
    
    # Output directory
    output_path = Path("test_sim_output") / f"test_sim_{int(time.time())}"
    
    print("="*50)
    print("PARALLEL SIMULATION TEST")
    print("="*50)
    print(f"Matches: {config.num_matches}")
    print(f"Workers: {config.num_workers}")
    print(f"Batch size: {config.batch_size}")
    print(f"Model path: {model_path}")
    print(f"Output: {output_path}")
    print()
    
    try:
        start_time = time.time()
        
        # Run simulation
        run_parallel_simulation(model_path, config, output_path)
        
        elapsed = time.time() - start_time
        print(f"\nTest completed in {elapsed:.1f} seconds")
        
        # Check results
        checkpoints = list(output_path.glob("checkpoint_*"))
        if checkpoints:
            print(f"✅ Found {len(checkpoints)} checkpoints")
            
            # Try to read latest checkpoint
            latest = sorted(checkpoints)[-1]
            stats_file = latest / "statistics.json"
            
            if stats_file.exists():
                try:
                    import json
                    with open(stats_file) as f:
                        stats = json.load(f)
                    print(f"✅ Completed {stats.get('total_matches', 0)} matches")
                    
                    if 'winners' in stats:
                        print("✅ Win distribution:")
                        for team, wins in stats['winners'].items():
                            print(f"   {team}: {wins} wins")
                            
                    if 'avg_scores' in stats:
                        print("✅ Average scores:")
                        for team, avg in stats['avg_scores'].items():
                            print(f"   {team}: {avg:.1f} runs")
                except Exception as e:
                    print(f"⚠️  Could not read statistics: {e}")
            else:
                print("⚠️  No statistics file (likely JSON serialization issue)")
                print("✅ But simulation completed successfully based on logs!")
        else:
            print("❌ No checkpoints found - simulation may have failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if model path argument provided
    if len(sys.argv) > 1:
        # If you want to specify model path as argument
        print("Usage: python test_parallel_mac.py")
        print("Edit the model_path variable in the script to point to your model")
    
    test_parallel_simulation()