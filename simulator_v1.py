import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

# Model setup (matching your training code)
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
EXTRA_BASE = 40
OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v: k for k, v in OUTCOME2TOK.items()}

@dataclass
class Player:
    name: str
    role: str  # "batsman", "bowler", "all-rounder"
    batting_position: int
    
@dataclass
class BallEvent:
    over: int
    ball: int
    bowler: str
    batsman: str
    outcome: str
    runs: int
    
@dataclass
class BatsmanStats:
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    strike_rate: float = 0.0
    
    def update(self, outcome: str):
        self.balls += 1
        if outcome == "4":
            self.runs += 4
            self.fours += 1
        elif outcome == "6":
            self.runs += 6
            self.sixes += 1
        elif outcome in ["1", "2", "3"]:
            self.runs += int(outcome)
        # Update strike rate
        self.strike_rate = (self.runs / self.balls * 100) if self.balls > 0 else 0.0

@dataclass
class BowlerStats:
    overs: float = 0.0
    runs: int = 0
    wickets: int = 0
    economy: float = 0.0
    balls: int = 0
    
    def update(self, outcome: str):
        self.balls += 1
        self.overs = self.balls // 6 + (self.balls % 6) * 0.1
        
        if outcome == "WICKET":
            self.wickets += 1
        elif outcome != "0":
            self.runs += int(outcome)
        
        # Update economy
        complete_overs = self.balls / 6
        self.economy = self.runs / complete_overs if complete_overs > 0 else 0.0

class MatchState:
    def __init__(self, team1_name: str, team2_name: str, team1_players: List[Player], 
                 team2_players: List[Player], venue: str = "Generic Stadium"):
        self.team1_name = team1_name
        self.team2_name = team2_name
        self.team1_players = team1_players
        self.team2_players = team2_players
        self.venue = venue
        
        # Match state
        self.innings = 1
        self.batting_team = team1_name
        self.bowling_team = team2_name
        self.current_batsmen = []  # [striker, non-striker]
        self.current_bowler = None
        self.score = 0
        self.wickets = 0
        self.overs = 0
        self.balls = 0
        
        # History
        self.recent_balls = []  # Last 5 balls
        self.ball_by_ball = []
        
        # Stats
        self.batting_stats = {team1_name: defaultdict(BatsmanStats), 
                             team2_name: defaultdict(BatsmanStats)}
        self.bowling_stats = {team1_name: defaultdict(BowlerStats), 
                             team2_name: defaultdict(BowlerStats)}
        
        # Partnership tracking
        self.partnership_runs = 0
        self.partnership_balls = 0
        
        # Results
        self.innings_scores = {}
        
    def get_batting_order(self) -> List[Player]:
        """Get batting team players sorted by batting position"""
        if self.batting_team == self.team1_name:
            return sorted(self.team1_players, key=lambda p: p.batting_position)
        else:
            return sorted(self.team2_players, key=lambda p: p.batting_position)
    
    def get_bowling_team_players(self) -> List[Player]:
        """Get bowling team players"""
        if self.bowling_team == self.team1_name:
            return [p for p in self.team1_players if p.role in ["bowler", "all-rounder"]]
        else:
            return [p for p in self.team2_players if p.role in ["bowler", "all-rounder"]]
    
    def initialize_innings(self):
        """Setup for new innings"""
        self.score = 0
        self.wickets = 0
        self.overs = 0
        self.balls = 0
        self.recent_balls = []
        self.partnership_runs = 0
        self.partnership_balls = 0
        
        # Get first two batsmen
        batting_order = self.get_batting_order()
        self.current_batsmen = [batting_order[0], batting_order[1]]
        
        # Select first bowler
        bowlers = self.get_bowling_team_players()
        self.current_bowler = random.choice(bowlers)
    
    def get_next_batsman(self) -> Optional[Player]:
        """Get next batsman in order"""
        batting_order = self.get_batting_order()
        current_names = [b.name for b in self.current_batsmen]
        
        for player in batting_order:
            stats = self.batting_stats[self.batting_team][player.name]
            # Player hasn't batted yet (no balls faced)
            if stats.balls == 0 and player.name not in current_names:
                return player
        return None
    
    def select_next_bowler(self) -> Player:
        """Select next bowler following cricket rules"""
        bowlers = self.get_bowling_team_players()
        available = []
        
        for bowler in bowlers:
            stats = self.bowling_stats[self.bowling_team][bowler.name]
            # Can't bowl consecutive overs
            if bowler != self.current_bowler:
                # Check over limit (max 4 overs in T20)
                if stats.overs < 4:
                    available.append(bowler)
        
        if not available:
            # Fallback: if no one available, pick anyone who can bowl more
            for bowler in bowlers:
                stats = self.bowling_stats[self.bowling_team][bowler.name]
                if stats.overs < 4 and bowler != self.current_bowler:
                    available.append(bowler)
        
        return random.choice(available) if available else random.choice(bowlers)
    
    def generate_prompt(self) -> str:
        """Generate prompt in the exact format from training"""
        # Format: "2.4: 12/1 | Recent: 4-W-1-0-0 | P:1@3b(2.0rr) | Wickramasinghe(Econ2.7) vs Samson(new,0 Runs @ 0 SR) | PP 2.4 | India vs Sri Lanka, PIC"
        
        # Current over.ball
        over_ball = f"{self.overs}.{self.balls % 6}"
        
        # Score
        score_str = f"{self.score}/{self.wickets}"
        
        # Recent balls (last 5)
        recent = "-".join(self.recent_balls[-5:]) if self.recent_balls else ""
        recent_str = f"Recent: {recent}" if recent else "Recent: "
        
        # Partnership
        partnership_rr = (self.partnership_runs / self.partnership_balls * 6) if self.partnership_balls > 0 else 0.0
        partnership_str = f"P:{self.wickets}@{self.partnership_balls}b({partnership_rr:.1f}rr)"
        
        # Bowler stats
        bowler_stats = self.bowling_stats[self.bowling_team][self.current_bowler.name]
        bowler_str = f"{self.current_bowler.name}(Econ{bowler_stats.economy:.1f})"
        
        # Batsman stats
        striker = self.current_batsmen[0]
        batsman_stats = self.batting_stats[self.batting_team][striker.name]
        
        if batsman_stats.balls == 0:
            batsman_str = f"{striker.name}(new,0 Runs @ 0 SR)"
        else:
            batsman_str = f"{striker.name}({batsman_stats.runs} Runs @ {batsman_stats.strike_rate:.0f} SR)"
        
        # Phase (PP = PowerPlay, etc)
        if self.overs < 6:
            phase = f"PP {over_ball}"
        elif self.overs < 16:
            phase = f"Mid {over_ball}"
        else:
            phase = f"Death {over_ball}"
        
        # Match info
        match_str = f"{self.batting_team} vs {self.bowling_team}, {self.venue[:3].upper()}"
        
        # Combine all parts
        prompt = f"{over_ball}: {score_str} | {recent_str} | {partnership_str} | {bowler_str} vs {batsman_str} | {phase} | {match_str}"
        
        return prompt
    
    def process_ball(self, outcome: str):
        """Update match state after a ball"""
        # Record ball
        self.balls += 1
        
        # Update recent balls
        if outcome == "WICKET":
            self.recent_balls.append("W")
        else:
            self.recent_balls.append(outcome)
        
        # Keep only last 5 balls
        if len(self.recent_balls) > 5:
            self.recent_balls.pop(0)
        
        # Update scores
        if outcome == "WICKET":
            self.wickets += 1
            # Reset partnership
            self.partnership_runs = 0
            self.partnership_balls = 0
            
            # Get new batsman
            new_batsman = self.get_next_batsman()
            if new_batsman:
                self.current_batsmen[0] = new_batsman  # New batsman on strike
        else:
            runs = int(outcome)
            self.score += runs
            self.partnership_runs += runs
            
            # Rotate strike on odd runs
            if runs % 2 == 1:
                self.current_batsmen[0], self.current_batsmen[1] = self.current_batsmen[1], self.current_batsmen[0]
        
        # Update partnership balls
        self.partnership_balls += 1
        
        # Update batsman stats
        striker = self.current_batsmen[0]
        self.batting_stats[self.batting_team][striker.name].update(outcome)
        
        # Update bowler stats
        self.bowling_stats[self.bowling_team][self.current_bowler.name].update(outcome)
        
        # Record ball event
        event = BallEvent(
            over=self.overs,
            ball=self.balls % 6,
            bowler=self.current_bowler.name,
            batsman=striker.name,
            outcome=outcome,
            runs=int(outcome) if outcome != "WICKET" else 0
        )
        self.ball_by_ball.append(event)
        
        # Check for over completion
        if self.balls % 6 == 0:
            self.overs = self.balls // 6
            # Rotate strike at end of over
            self.current_batsmen[0], self.current_batsmen[1] = self.current_batsmen[1], self.current_batsmen[0]
            # Select new bowler
            self.current_bowler = self.select_next_bowler()
        else:
            self.overs = self.balls // 6
    
    def is_innings_complete(self) -> bool:
        """Check if innings is complete"""
        return self.wickets >= 10 or self.overs >= 20
    
    def switch_innings(self):
        """Switch to second innings"""
        # Store first innings score
        self.innings_scores[self.batting_team] = {"runs": self.score, "wickets": self.wickets, "overs": self.overs}
        
        # Swap teams
        self.innings = 2
        self.batting_team, self.bowling_team = self.bowling_team, self.batting_team
        
        # Initialize second innings
        self.initialize_innings()

def predict_outcome(prompt: str, model, tokenizer, device) -> tuple[str, dict]:
    """Get outcome from model - returns outcome and probability distribution"""
    model.eval()
    prompt_with_space = prompt.rstrip() + " "
    inputs = tokenizer(prompt_with_space, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Disable autocast for MPS due to numerical instability
    if device.type == 'mps':
        with torch.no_grad():
            outputs = model(**inputs)
    else:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(**inputs)
    
    logits = outputs.logits
    last_token_logits = logits[0, -1, :]
    
    # Get token IDs for outcomes
    outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
    
    # Debug: Check if tokens exist
    if any(tid == tokenizer.unk_token_id for tid in outcome_token_ids):
        print("Warning: Some outcome tokens not found in vocabulary!")
        print("Outcome tokens:", list(OUTCOME2TOK.values()))
        print("Token IDs:", outcome_token_ids)
    
    # Extract logits for outcome tokens
    outcome_logits = last_token_logits[outcome_token_ids]
    
    # Debug: Check for extreme values
    if torch.isinf(outcome_logits).any() or torch.isnan(outcome_logits).any():
        print("Warning: Found inf/nan in outcome logits!")
        print("Logits range:", outcome_logits.min().item(), "to", outcome_logits.max().item())
        print("Sample prompt:", prompt_with_space[:100])
        # Use uniform distribution as fallback
        outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
    else:
        # Use log-softmax for numerical stability
        log_probs = torch.log_softmax(outcome_logits, dim=-1)
        outcome_probs = torch.exp(log_probs).cpu().numpy()
        
        # Check for NaN in probabilities
        if np.isnan(outcome_probs).any():
            print("Warning: NaN in probabilities, using uniform distribution")
            outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
    
    # Ensure probabilities sum to 1 and are positive
    outcome_probs = np.clip(outcome_probs, 1e-8, 1.0)
    outcome_probs = outcome_probs / outcome_probs.sum()
    
    # Create probability dictionary
    prob_dict = {outcome: prob for outcome, prob in zip(OUTCOMES, outcome_probs)}
    
    # Sample from distribution
    outcome_idx = np.random.choice(len(OUTCOMES), p=outcome_probs)
    selected_outcome = OUTCOMES[outcome_idx]
    
    return selected_outcome, prob_dict

def print_ball_commentary(match_state, prompt, outcome, prob_dict, ball_number):
    """Print ball-by-ball commentary with probabilities"""
    print(f"\n{'='*80}")
    print(f"Ball #{ball_number} | {match_state.overs}.{match_state.balls % 6}")
    print(f"{'='*80}")
    
    # Current match situation
    print(f"🏏 {match_state.batting_team}: {match_state.score}/{match_state.wickets}")
    print(f"🎯 {match_state.current_bowler.name} to {match_state.current_batsmen[0].name}")
    
    if match_state.innings == 2:
        target = match_state.innings_scores[match_state.team1_name]["runs"] + 1
        need = target - match_state.score
        balls_left = (20 - match_state.overs) * 6 - (match_state.balls % 6)
        req_rate = (need / balls_left * 6) if balls_left > 0 else 0
        print(f"🎯 Need {need} runs in {balls_left} balls (RRR: {req_rate:.2f})")
    
    # Recent balls
    if match_state.recent_balls:
        print(f"📊 Recent: {'-'.join(match_state.recent_balls[-5:])}")
    
    # Model predictions
    print(f"\n🤖 Model Input: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"\n📈 Probabilities:")
    
    # Sort probabilities for better display
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    for outcome_name, prob in sorted_probs:
        bar_length = int(prob * 30)  # Scale to 30 characters
        bar = '█' * bar_length + '░' * (30 - bar_length)
        star = ' ⭐' if outcome_name == outcome else ''
        print(f"  {outcome_name:6s}: {prob:.3f} |{bar}|{star}")
    
    # Selected outcome
    print(f"\n🎲 Selected: {outcome}")
    
    # Outcome description
    if outcome == "WICKET":
        print(f"🚨 WICKET! {match_state.current_batsmen[0].name} is OUT!")
    elif outcome == "6":
        print(f"🚀 SIX! {match_state.current_batsmen[0].name} sends it sailing!")
    elif outcome == "4":
        print(f"🏏 FOUR! Beautiful shot by {match_state.current_batsmen[0].name}!")
    elif outcome in ["1", "2", "3"]:
        print(f"🏃 {outcome} run(s) taken")
    else:  # outcome == "0"
        print(f"⚫ Dot ball")

def simulate_match(model, tokenizer, device, team1_name: str, team2_name: str, 
                  team1_players: List[Player], team2_players: List[Player], 
                  venue: str = "Generic Stadium", verbose: bool = True, pause_balls: bool = False) -> Dict:
    """Simulate a complete T20 match
    
    Args:
        verbose: Print ball-by-ball commentary
        pause_balls: Pause after each ball for manual progression (press Enter)
    """
    """Simulate a complete T20 match"""
    
    match = MatchState(team1_name, team2_name, team1_players, team2_players, venue)
    ball_count = 0
    
    # First innings
    if verbose:
        print(f"\n🏏 FIRST INNINGS")
        print(f"🏏 {match.batting_team} batting vs {match.bowling_team}")
        print(f"🏟️  Venue: {venue}")
    
    match.initialize_innings()
    while not match.is_innings_complete():
        ball_count += 1
        prompt = match.generate_prompt()
        outcome, prob_dict = predict_outcome(prompt, model, tokenizer, device)
        
        if verbose:
            print_ball_commentary(match, prompt, outcome, prob_dict, ball_count)
            if pause_balls:
                input("Press Enter for next ball...")
        
        match.process_ball(outcome)
        
        if verbose and match.is_innings_complete():
            print(f"\n🏁 END OF FIRST INNINGS")
            print(f"🏏 {match.batting_team}: {match.score}/{match.wickets} ({match.overs} overs)")
    
    # Second innings
    match.switch_innings()
    target = match.innings_scores[team1_name]["runs"] + 1
    
    if verbose:
        print(f"\n🏏 SECOND INNINGS")
        print(f"🏏 {match.batting_team} batting vs {match.bowling_team}")
        print(f"🎯 Target: {target} runs")
    
    while not match.is_innings_complete() and match.score < target:
        ball_count += 1
        prompt = match.generate_prompt()
        outcome, prob_dict = predict_outcome(prompt, model, tokenizer, device)
        
        if verbose:
            print_ball_commentary(match, prompt, outcome, prob_dict, ball_count)
            if pause_balls:
                input("Press Enter for next ball...")
        
        match.process_ball(outcome)
        
        if match.score >= target:
            if verbose:
                print(f"\n🎉 {match.batting_team} WINS!")
            break
    
    if verbose and match.score < target:
        print(f"\n🏁 END OF SECOND INNINGS")
        print(f"🏏 {match.batting_team}: {match.score}/{match.wickets} ({match.overs} overs)")
    
    # Store second innings score
    match.innings_scores[match.batting_team] = {"runs": match.score, "wickets": match.wickets, "overs": match.overs}
    
    # Determine winner
    if match.innings_scores[team2_name]["runs"] >= target:
        winner = team2_name
        margin = f"by {10 - match.innings_scores[team2_name]['wickets']} wickets"
    else:
        winner = team1_name
        margin = f"by {target - match.innings_scores[team2_name]['runs'] - 1} runs"
    
    return {
        "winner": winner,
        "margin": margin,
        "innings_scores": match.innings_scores,
        "batting_stats": dict(match.batting_stats),
        "bowling_stats": dict(match.bowling_stats),
        "ball_by_ball": match.ball_by_ball
    }

def get_optimal_device():
    """Get best available device for Apple Silicon"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_path: str, base_model_name: str = "Qwen/Qwen1.5-1.8B"):
    """Load fine-tuned model - optimized for Apple Silicon"""
    device = get_optimal_device()
    print(f"Using device: {device}")
    
    # Setup tokenizer (matching verification script)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Add special tokens
    special_tokens_to_add = list(OUTCOME2TOK.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    
    # Load base model with device-specific optimizations
    # Use float32 for MPS due to numerical stability issues with float16
    if device.type == 'mps':
        print("Using float32 for MPS numerical stability")
        dtype = torch.float32
    else:
        dtype = torch.float16
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=None,  # Manual device placement
        trust_remote_code=False,
        low_cpu_mem_usage=True
    )
    
    # Resize embeddings for special tokens (crucial step)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)
    model.eval()
    
    # Skip torch.compile for MPS due to Metal shader compilation issues
    # Only attempt compilation for CUDA
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled for optimization")
        except:
            print("Torch compile failed, using standard model")
    else:
        print("Using standard model (MPS doesn't support torch.compile reliably)")
    
    return model, tokenizer, device

def create_sample_teams():
    """Create sample teams for testing"""
    team1_players = [
        Player("Rohit", "batsman", 1),
        Player("Gill", "batsman", 2),
        Player("Kohli", "batsman", 3),
        Player("Iyer", "batsman", 4),
        Player("Rahul", "batsman", 5),
        Player("Hardik", "all-rounder", 6),
        Player("Jadeja", "all-rounder", 7),
        Player("Ashwin", "bowler", 8),
        Player("Bumrah", "bowler", 9),
        Player("Siraj", "bowler", 10),
        Player("Shami", "bowler", 11),
    ]
    
    team2_players = [
        Player("Nissanka", "batsman", 1),
        Player("Mendis", "batsman", 2),
        Player("Asalanka", "batsman", 3),
        Player("Dhananjaya", "all-rounder", 4),
        Player("Chandimal", "batsman", 5),
        Player("Hasaranga", "all-rounder", 6),
        Player("Shanaka", "all-rounder", 7),
        Player("Chamika", "all-rounder", 8),
        Player("Theekshana", "bowler", 9),
        Player("Madushanka", "bowler", 10),
        Player("Rajitha", "bowler", 11),
    ]
    
    return team1_players, team2_players

if __name__ == "__main__":
    # Example usage
    model_path = "models/checkpoint-24000"  # Your actual checkpoint path
    
    print("Loading model...")
    model, tokenizer, device = load_model(model_path)
    
    print("Creating teams...")
    team1_players, team2_players = create_sample_teams()
    
    print("Simulating match...")
    result = simulate_match(
        model, tokenizer, device,
        "India", "Sri Lanka",
        team1_players, team2_players,
        "Premadasa International Cricket Stadium",
        verbose=True,  # Set to False for quiet simulation
        pause_balls=False  # Set to True to pause after each ball
    )
    
    # Print scorecard
    print("\n=== MATCH RESULT ===")
    print(f"Winner: {result['winner']} {result['margin']}")
    print("\n=== INNINGS SUMMARY ===")
    for team, score in result['innings_scores'].items():
        print(f"{team}: {score['runs']}/{score['wickets']} ({score['overs']} overs)")
    
    print("\n=== TOP BATSMEN ===")
    for team in ["India", "Sri Lanka"]:
        print(f"\n{team}:")
        batsmen = [(name, stats) for name, stats in result['batting_stats'][team].items() if stats.balls > 0]
        batsmen.sort(key=lambda x: x[1].runs, reverse=True)
        for name, stats in batsmen[:5]:
            print(f"  {name}: {stats.runs} ({stats.balls}b, {stats.fours}x4, {stats.sixes}x6) SR: {stats.strike_rate:.1f}")
    
    print("\n=== BOWLING FIGURES ===")
    for team in ["India", "Sri Lanka"]:
        print(f"\n{team}:")
        bowlers = [(name, stats) for name, stats in result['bowling_stats'][team].items() if stats.balls > 0]
        bowlers.sort(key=lambda x: x[1].wickets, reverse=True)
        for name, stats in bowlers:
            print(f"  {name}: {stats.wickets}/{stats.runs} ({stats.overs} ov) Econ: {stats.economy:.1f}")