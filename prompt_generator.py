import json
import os
from pathlib import Path

def map_outcome(runs, is_wicket):
    """Map ball outcome to our token categories"""
    if is_wicket:
        return "WICKET"
    
    # Map runs to our categories
    if runs >= 6:
        return "6"
    elif runs == 5:
        return "4"  # Map 5s to 4s as specified
    else:
        return str(runs)

def calculate_game_state(innings_data, current_over_idx, current_delivery_idx):
    """Calculate current score, wickets, and ball count"""
    total_score = 0
    total_wickets = 0
    total_balls = 0
    
    # Count up to current delivery
    for over_idx, over in enumerate(innings_data['overs']):
        if over_idx > current_over_idx:
            break
            
        for delivery_idx, delivery in enumerate(over['deliveries']):
            if over_idx == current_over_idx and delivery_idx >= current_delivery_idx:
                break
                
            total_score += delivery['runs']['total']
            if 'wickets' in delivery:
                total_wickets += len(delivery['wickets'])
            
            # Count balls (excluding wides and no-balls)
            is_extra = 'extras' in delivery
            is_wide_or_noball = False
            if is_extra:
                extras = delivery['extras']
                is_wide_or_noball = 'wides' in extras or 'noballs' in extras
            
            if not is_wide_or_noball:
                total_balls += 1
    
    return total_score, total_wickets, total_balls

def format_over_ball(total_balls):
    """Convert total balls to over.ball format"""
    if total_balls == 0:
        return "0.1"
    
    overs = total_balls // 6
    balls = total_balls % 6
    return f"{overs}.{balls + 1}"

def parse_match_to_prompts(json_file_path):
    """Parse a single match JSON and return list of prompts"""
    with open(json_file_path, 'r') as f:
        match_data = json.load(f)
    
    prompts = []
    match_info = match_data['info']
    
    # Extract match context
    venue = match_info['venue']
    city = match_info['city']
    date = match_info['dates'][0]
    teams = match_info['teams']
    season = match_info.get('season', '')
    toss_winner = match_info['toss']['winner']
    toss_decision = match_info['toss']['decision']
    
    # Process each innings
    for innings_idx, innings in enumerate(match_data['innings'], 1):
        team = innings['team']
        target = None
        
        # Get target for second innings
        if 'target' in innings:
            target = innings['target']['runs']
        
        # Process each delivery
        for over_idx, over in enumerate(innings['overs']):
            for delivery_idx, delivery in enumerate(over['deliveries']):
                
                # Calculate current game state
                score, wickets, balls = calculate_game_state(innings, over_idx, delivery_idx)
                over_ball = format_over_ball(balls)
                
                # Extract delivery details
                bowler = delivery['bowler']
                batter = delivery['batter']
                non_striker = delivery['non_striker']
                
                # Determine outcome
                runs = delivery['runs']['total']
                is_wicket = 'wickets' in delivery
                outcome = map_outcome(runs, is_wicket)
                
                # Create prompt
                innings_suffix = "1st" if innings_idx == 1 else "2nd"
                
                if innings_idx == 2 and target:
                    # Second innings with target
                    prompt = f"{bowler} bowling to {batter}, {score}/{wickets} after {over_ball} overs, {innings_suffix} innings, chasing {target}, {venue}, {date}"
                else:
                    # First innings or no target
                    prompt = f"{bowler} bowling to {batter}, {score}/{wickets} after {over_ball} overs, {innings_suffix} innings, {venue}, {date}"
                
                prompts.append({
                    "prompt": prompt,
                    "target": outcome
                })
    
    return prompts

def parse_folder_to_jsonl(folder_path, output_file='cricket_prompts.jsonl'):
    """Parse all JSON files in folder and create JSONL"""
    folder = Path(folder_path)
    json_files = list(folder.glob('*.json'))
    
    total_prompts = 0
    outcome_counts = {}
    
    with open(output_file, 'w') as out_f:
        for json_file in json_files:
            try:
                print(f"Processing {json_file.name}...")
                match_prompts = parse_match_to_prompts(json_file)
                
                for prompt_data in match_prompts:
                    # Write to JSONL
                    out_f.write(json.dumps(prompt_data) + '\n')
                    
                    # Count outcomes
                    outcome = prompt_data['target']
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                    total_prompts += 1
                
                print(f"  Generated {len(match_prompts)} prompts")
                
            except Exception as e:
                print(f"Error processing {json_file.name}: {e}")
    
    print(f"\nTotal prompts generated: {total_prompts}")
    print(f"Output saved to: {output_file}")
    print("\nOutcome distribution:")
    for outcome, count in sorted(outcome_counts.items()):
        percentage = (count / total_prompts) * 100
        print(f"  {outcome}: {count} ({percentage:.2f}%)")
    
    return total_prompts, outcome_counts

# Example usage and testing
if __name__ == "__main__":
    # Test with single file first
    '''
    try:
        sample_prompts = parse_match_to_prompts('data/226374.json')
        
        print("Sample prompts from the match:")
        
        # Show first 5 prompts from first innings
        print("\n=== FIRST INNINGS SAMPLES ===")
        first_innings_prompts = [p for p in sample_prompts if "1st innings" in p['prompt']]
        for i, prompt in enumerate(first_innings_prompts[:5]):
            print(f"{i+1}. \"{prompt['prompt']}\"")
            print(f"   Target: \"{prompt['target']}\"")
            print()
        
        # Show first 3 prompts from second innings
        print("=== SECOND INNINGS SAMPLES ===")
        second_innings_prompts = [p for p in sample_prompts if "2nd innings" in p['prompt']]
        for i, prompt in enumerate(second_innings_prompts[:3]):
            print(f"{i+1}. \"{prompt['prompt']}\"")
            print(f"   Target: \"{prompt['target']}\"")
            print()
        
        print(f"Total prompts from sample file: {len(sample_prompts)}")
        
        # Show outcome distribution
        outcome_counts = {}
        for prompt in sample_prompts:
            outcome = prompt['target']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        print("\nOutcome distribution:")
        total = len(sample_prompts)
        for outcome in ["0", "1", "2", "3", "4", "6", "WICKET"]:
            count = outcome_counts.get(outcome, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {outcome}: {count} ({percentage:.1f}%)")
        
        # Save sample to JSONL for inspection
        with open('sample_prompts.jsonl', 'w') as f:
            for prompt in sample_prompts[:20]:  # First 20 prompts
                f.write(json.dumps(prompt) + '\n')
        print(f"\nSaved first 20 prompts to sample_prompts.jsonl")
        
    except FileNotFoundError:
        print("Sample file '226374.json' not found. Please ensure it's in the current directory.")
        print("To process a folder of JSON files, use:")
        print("parse_folder_to_jsonl('/path/to/your/json/files/')")
    '''
    # To process entire folder, uncomment and modify path:
    parse_folder_to_jsonl('/Users/aryamangupta/CricML/llm-finetune/data', 'cricket_prompts.jsonl')