import json
import os
from pathlib import Path
from collections import defaultdict

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

def get_match_phase(current_over, is_powerplay):
    """Determine match phase based on over number and powerplay status"""
    if is_powerplay:
        return "PP"  # PowerPlay
    elif current_over >= 16:
        return "Death"  # Death overs (17-20)
    else:
        return "Mid"  # Middle overs

def is_in_powerplay(current_over, current_ball, powerplays):
    """Check if current delivery is in powerplay"""
    if not powerplays:
        return False
    
    # Convert current position to decimal format (like 5.3 for 5th over, 3rd ball)
    current_pos = current_over + (current_ball / 6)
    
    for pp in powerplays:
        if pp['from'] <= current_pos <= pp['to']:
            return True
    return False

def calculate_required_run_rate(target, current_score, overs_remaining):
    """Calculate required run rate for 2nd innings"""
    if overs_remaining <= 0:
        return 0
    runs_needed = target - current_score
    return round(runs_needed / overs_remaining, 1)

def get_recent_balls(innings_data, current_over_idx, current_delivery_idx, count=5):
    """Get last N legal deliveries before current position"""
    recent_balls = []
    
    # Go backwards through all deliveries to find last N legal ones
    for over_idx in range(current_over_idx, -1, -1):
        if over_idx >= len(innings_data['overs']):
            continue
            
        over = innings_data['overs'][over_idx]
        
        # Determine delivery range for this over
        if over_idx == current_over_idx:
            # Current over: only deliveries before current one
            delivery_range = range(current_delivery_idx - 1, -1, -1)
        else:
            # Previous overs: all deliveries
            delivery_range = range(len(over['deliveries']) - 1, -1, -1)
        
        for delivery_idx in delivery_range:
            if len(recent_balls) >= count:
                break
                
            delivery = over['deliveries'][delivery_idx]
            
            # Check if it's a legal delivery (not wide or no-ball)
            is_extra = 'extras' in delivery
            is_wide_or_noball = False
            if is_extra:
                extras = delivery['extras']
                is_wide_or_noball = 'wides' in extras or 'noballs' in extras
            
            if not is_wide_or_noball:
                runs = delivery['runs']['total']
                if 'wickets' in delivery:
                    recent_balls.append('W')
                else:
                    recent_balls.append(str(runs))
        
        if len(recent_balls) >= count:
            break
    
    # Reverse to get chronological order and pad if needed
    recent_balls.reverse()
    while len(recent_balls) < count:
        recent_balls.insert(0, '-')
    
    return recent_balls[-count:]  # Return last N balls

def get_partnership_stats(innings_data, current_over_idx, current_delivery_idx):
    """Get partnership runs and balls since last wicket"""
    partnership_runs = 0
    partnership_balls = 0
    
    # Find the last wicket position
    last_wicket_over = -1
    last_wicket_delivery = -1
    
    # Search backwards for last wicket
    for over_idx in range(current_over_idx, -1, -1):
        if over_idx >= len(innings_data['overs']):
            continue
            
        over = innings_data['overs'][over_idx]
        
        # Determine delivery range for this over
        if over_idx == current_over_idx:
            delivery_range = range(current_delivery_idx - 1, -1, -1)
        else:
            delivery_range = range(len(over['deliveries']) - 1, -1, -1)
        
        for delivery_idx in delivery_range:
            delivery = over['deliveries'][delivery_idx]
            if 'wickets' in delivery:
                last_wicket_over = over_idx
                last_wicket_delivery = delivery_idx
                break
        
        if last_wicket_over != -1:
            break
    
    # Calculate partnership stats from after last wicket
    start_over = last_wicket_over if last_wicket_over != -1 else 0
    start_delivery = last_wicket_delivery + 1 if last_wicket_over != -1 else 0
    
    for over_idx in range(start_over, current_over_idx + 1):
        if over_idx >= len(innings_data['overs']):
            continue
            
        over = innings_data['overs'][over_idx]
        
        # Determine delivery range for this over
        if over_idx == start_over and over_idx == current_over_idx:
            # Same over as start and current
            delivery_range = range(start_delivery, current_delivery_idx)
        elif over_idx == start_over:
            # Starting over
            delivery_range = range(start_delivery, len(over['deliveries']))
        elif over_idx == current_over_idx:
            # Current over
            delivery_range = range(0, current_delivery_idx)
        else:
            # Middle overs
            delivery_range = range(0, len(over['deliveries']))
        
        for delivery_idx in delivery_range:
            delivery = over['deliveries'][delivery_idx]
            partnership_runs += delivery['runs']['total']
            
            # Count legal deliveries only
            is_extra = 'extras' in delivery
            is_wide_or_noball = False
            if is_extra:
                extras = delivery['extras']
                is_wide_or_noball = 'wides' in extras or 'noballs' in extras
            
            if not is_wide_or_noball:
                partnership_balls += 1
    
    return partnership_runs, partnership_balls

class PlayerStatsTracker:
    """Track running statistics for players during the match"""
    
    def __init__(self):
        self.batter_stats = defaultdict(lambda: {'runs': 0, 'balls': 0, 'recent_balls': []})
        self.bowler_stats = defaultdict(lambda: {'runs_conceded': 0, 'balls_bowled': 0, 'wickets': 0, 'recent_balls': []})
    
    def update_batter_stats(self, batter, runs, is_wicket):
        """Update batter statistics"""
        stats = self.batter_stats[batter]
        stats['runs'] += runs  # Use runs.batter, not runs.total
        stats['balls'] += 1
        stats['recent_balls'].append(runs)
        
        # Keep only last 20 balls for recent form
        if len(stats['recent_balls']) > 20:
            stats['recent_balls'] = stats['recent_balls'][-20:]
    
    def update_bowler_stats(self, bowler, runs_conceded, is_wicket, is_legal_delivery):
        """Update bowler statistics"""
        stats = self.bowler_stats[bowler]
        stats['runs_conceded'] += runs_conceded
        if is_legal_delivery:
            stats['balls_bowled'] += 1
        if is_wicket:
            stats['wickets'] += 1
        
        stats['recent_balls'].append(runs_conceded)
        
        # Keep only last 24 balls for recent form
        if len(stats['recent_balls']) > 24:
            stats['recent_balls'] = stats['recent_balls'][-24:]
    
    def get_batter_form(self, batter):
        """Get batter's current form (verbose format)"""
        stats = self.batter_stats[batter]
        if stats['balls'] == 0:
            return "New"
        
        runs = stats['runs']
        balls = stats['balls']
        strike_rate = round((runs / balls) * 100, 0)
        return f"{runs} Runs @ {int(strike_rate)} SR"
    
    def get_bowler_form(self, bowler):
        """Get bowler's current form (verbose format)"""
        stats = self.bowler_stats[bowler]
        if stats['balls_bowled'] == 0:
            return "New"
        
        overs = stats['balls_bowled'] / 6.0
        economy = round(stats['runs_conceded'] / overs, 1) if overs > 0 else 0
        wickets = stats['wickets']
        
        if wickets > 0:
            return f"Econ{economy},{wickets}W"
        else:
            return f"Econ{economy}"
    
    def get_player_status(self, player):
        """Get player status: new vs set"""
        stats = self.batter_stats[player]
        if stats['balls'] < 10:
            return "new"
        else:
            return "set"

def calculate_game_state_enhanced(innings_data, current_over_idx, current_delivery_idx, powerplays):
    """Calculate enhanced game state (score, wickets, balls only)"""
    total_score = 0
    total_wickets = 0
    total_balls = 0
    
    # Count up to current delivery (but not including it)
    for over_idx, over in enumerate(innings_data['overs']):
        if over_idx > current_over_idx:
            break
            
        for delivery_idx, delivery in enumerate(over['deliveries']):
            if over_idx == current_over_idx and delivery_idx >= current_delivery_idx:
                break
            
            runs = delivery['runs']['total']
            is_wicket = 'wickets' in delivery
            
            total_score += runs
            if is_wicket:
                total_wickets += len(delivery['wickets'])
            
            # Check if it's a legal delivery (not wide or no-ball)
            is_extra = 'extras' in delivery
            is_wide_or_noball = False
            if is_extra:
                extras = delivery['extras']
                is_wide_or_noball = 'wides' in extras or 'noballs' in extras
            
            if not is_wide_or_noball:
                total_balls += 1
    
    # Determine current over and ball
    current_over = total_balls // 6
    current_ball = (total_balls % 6) + 1
    
    # Check powerplay status
    in_powerplay = is_in_powerplay(current_over, current_ball, powerplays)
    
    return {
        'score': total_score,
        'wickets': total_wickets,
        'balls': total_balls,
        'over': current_over,
        'ball': current_ball,
        'in_powerplay': in_powerplay
    }

def format_over_ball(total_balls):
    """Convert total balls to over.ball format"""
    if total_balls == 0:
        return "0.1"
    
    overs = total_balls // 6
    balls = (total_balls % 6) + 1
    return f"{overs}.{balls}"

def parse_match_to_prompts_enhanced(json_file_path):
    """Parse a single match JSON and return enhanced prompts with consistent structure"""
    with open(json_file_path, 'r') as f:
        match_data = json.load(f)
    
    prompts = []
    match_info = match_data['info']
    
    # Extract match context
    venue = match_info['venue'].split(',')[0]  # Shortened venue name
    date = match_info['dates'][0]
    teams = match_info['teams']
    
    # Create a shorter venue code if possible
    venue_short = venue.replace(' Cricket Ground', '').replace(' Stadium', '')
    if len(venue_short) > 15:
        # Create abbreviation for very long names
        words = venue_short.split()
        if len(words) > 1:
            venue_short = ''.join([word[0].upper() for word in words[:3]])
    
    first_innings_score = None
    
    # Process each innings
    for innings_idx, innings in enumerate(match_data['innings'], 1):
        team = innings['team']
        target = None
        powerplays = innings.get('powerplays', [])
        
        # Initialize stats tracker for this innings
        stats_tracker = PlayerStatsTracker()
        
        # Get target for second innings
        if 'target' in innings:
            target = innings['target']['runs']
        
        # Process each delivery
        for over_idx, over in enumerate(innings['overs']):
            for delivery_idx, delivery in enumerate(over['deliveries']):
                
                # Calculate enhanced game state (up to but not including current delivery)
                game_state = calculate_game_state_enhanced(
                    innings, over_idx, delivery_idx, powerplays
                )
                
                # Extract delivery details
                bowler = delivery['bowler']
                batter = delivery['batter']
                
                # Get enhanced context BEFORE updating stats
                recent_balls = get_recent_balls(innings, over_idx, delivery_idx, 5)
                partnership_runs, partnership_balls = get_partnership_stats(innings, over_idx, delivery_idx)
                
                # Get player forms and status BEFORE updating with current delivery
                batter_form = stats_tracker.get_batter_form(batter)
                bowler_form = stats_tracker.get_bowler_form(bowler)
                batter_status = stats_tracker.get_player_status(batter)
                
                # Determine outcome
                runs = delivery['runs']['total']
                batter_runs = delivery['runs']['batter']  # Use batter-specific runs
                is_wicket = 'wickets' in delivery
                outcome = map_outcome(runs, is_wicket)
                
                # Determine match phase
                match_phase = get_match_phase(game_state['over'], game_state['in_powerplay'])
                
                # Calculate overs remaining and required run rate for 2nd innings
                overs_remaining = 20 - (game_state['balls'] / 6)
                
                # CREATE ENHANCED PROMPT - VERBOSE 25-TOKEN FORMAT
                # Format: Situation | Recent | Partnership | Players | Context
                
                # 1. SITUATION (5-6 tokens)
                over_ball = format_over_ball(game_state['balls'])
                if innings_idx == 2 and target:
                    rrr = calculate_required_run_rate(target, game_state['score'], overs_remaining)
                    runs_needed = target - game_state['score']
                    situation = f"{over_ball}: {game_state['score']}/{game_state['wickets']} Need{runs_needed}@{rrr}"
                else:
                    situation = f"{over_ball}: {game_state['score']}/{game_state['wickets']}"
                
                # 2. RECENT MOMENTUM (5-6 tokens)
                recent_str = "-".join(recent_balls)
                recent = f"Recent: {recent_str}"
                
                # 3. PARTNERSHIP WITH RUN RATE (4 tokens)
                if partnership_balls > 0:
                    p_rate = round((partnership_runs / partnership_balls) * 6, 1)
                    partnership = f"P:{partnership_runs}@{partnership_balls}b({p_rate}rr)"
                else:
                    partnership = "P:0@0b(0rr)"
                
                # 4. ENHANCED PLAYERS (8-10 tokens)
                # Use shortened names for space efficiency
                bowler_short = bowler.split()[-1] if " " in bowler else bowler  # Last name only
                batter_short = batter.split()[-1] if " " in batter else batter   # Last name only
                
                # Show batter status and verbose form
                if batter_form == "New":
                    batter_display = f"{batter_short}({batter_status})"
                else:
                    batter_display = f"{batter_short}({batter_status},{batter_form})"
                
                players = f"{bowler_short}({bowler_form}) vs {batter_display}"
                
                # 5. ENHANCED CONTEXT (4-5 tokens)
                # Show current over with phase, plus consistent venue
                current_over_display = f"{game_state['over']}.{game_state['ball']}"
                
                if match_phase == "PP":
                    context = f"PP {current_over_display} | {teams[0]} vs {teams[1]}, {venue_short}"
                elif match_phase == "Death":
                    context = f"Death {current_over_display} | {teams[0]} vs {teams[1]}, {venue_short}"
                else:
                    context = f"{teams[0]} vs {teams[1]}, {venue_short}"
                
                # Combine all parts with consistent separator
                prompt = f"{situation} | {recent} | {partnership} | {players} | {context}"
                
                prompts.append({
                    "prompt": prompt,
                    "target": outcome
                })
                
                # NOW update statistics AFTER creating the prompt
                # Check if it's a legal delivery (not wide or no-ball)
                is_extra = 'extras' in delivery
                is_wide_or_noball = False
                if is_extra:
                    extras = delivery['extras']
                    is_wide_or_noball = 'wides' in extras or 'noballs' in extras
                
                is_legal_delivery = not is_wide_or_noball
                
                # Update statistics for next iteration
                stats_tracker.update_batter_stats(batter, batter_runs, is_wicket)
                stats_tracker.update_bowler_stats(bowler, runs, is_wicket, is_legal_delivery)
        
        # Store first innings score for target calculation
        if innings_idx == 1:
            first_innings_score = game_state['score']
    
    return prompts

def parse_folder_to_jsonl_enhanced(folder_path, output_file='cricket_prompts_phase2_enhanced.jsonl'):
    """Parse all JSON files in folder and create enhanced JSONL"""
    folder = Path(folder_path)
    json_files = list(folder.glob('*.json'))
    
    total_prompts = 0
    outcome_counts = {}
    
    with open(output_file, 'w') as out_f:
        for json_file in json_files:
            try:
                print(f"Processing {json_file.name}...")
                match_prompts = parse_match_to_prompts_enhanced(json_file)
                
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
        print("=== TESTING PHASE 2 ENHANCED PARSER (25 TOKENS) ===")
        sample_prompts = parse_match_to_prompts_enhanced('data/226374.json')
        
        print("Phase 2 Enhanced prompts (verbose 25-token format):")
        
        # Show first 10 prompts from first innings
        print("\n=== FIRST INNINGS SAMPLES ===")
        first_innings_prompts = [p for p in sample_prompts if "Need" not in p['prompt']]
        for i, prompt in enumerate(first_innings_prompts[:10]):
            print(f"{i+1}. \"{prompt['prompt']}\"")
            print(f"   Target: \"{prompt['target']}\"")
            print()
        
        # Show first 6 prompts from second innings
        print("=== SECOND INNINGS SAMPLES ===")
        second_innings_prompts = [p for p in sample_prompts if "Need" in p['prompt']]
        for i, prompt in enumerate(second_innings_prompts[:6]):
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
        
        # Analyze token count
        sample_prompt = first_innings_prompts[5]['prompt'] if len(first_innings_prompts) > 5 else first_innings_prompts[0]['prompt']
        token_count = len(sample_prompt.split())
        print(f"\nSample token count: {token_count} tokens")
        print(f"Sample prompt: \"{sample_prompt}\"")
        
        # Save sample to JSONL for inspection
        with open('phase2_sample_prompts.jsonl', 'w') as f:
            for prompt in sample_prompts[:30]:  # First 30 prompts
                f.write(json.dumps(prompt) + '\n')
        print(f"\nSaved first 30 Phase 2 prompts to phase2_sample_prompts.jsonl")
        
        # Show Phase 1 vs Phase 2 comparison
        print("\n=== PHASE 1 vs PHASE 2 COMPARISON ===")
        print("PHASE 1 FORMAT (18 tokens):")
        print("  \"1.3: 6/0 PP | GJP Kruger(E2.0) vs DR Martyn(4*@400) | Australia vs South Africa, Brisbane\"")
        print("\nPHASE 2 FORMAT (~25 tokens):")
        print(f"  \"{sample_prompt}\"")
        print("\nPhase 2 Enhancements:")
        print("  ✓ Recent balls momentum (Recent: X-X-X)")
        print("  ✓ Partnership context (P:XrXb)")
        print("  ✓ Player status (new/set)")
        print("  ✓ Consistent structure across all prompts")
        print("  ✓ Enhanced context density")
        
    except FileNotFoundError:
        print("Sample file '226374.json' not found. Please ensure it's in the current directory.")
        print("To process a folder of JSON files, use:")
        print("parse_folder_to_jsonl_enhanced('/path/to/your/json/files/')")
    '''
    # To process entire folder, uncomment and modify path:
    parse_folder_to_jsonl_enhanced('/Users/aryamangupta/CricML/llm-finetune/data', 'cricket_prompts-v2.jsonl')