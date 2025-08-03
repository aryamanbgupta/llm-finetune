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
        
        # Calculate overs in cricket format (0.1, 0.2... 0.5, 1.0, 1.1...)
        complete_overs = self.balls // 6
        balls_in_over = self.balls % 6
        self.overs = complete_overs + (balls_in_over * 0.1)
        
        if outcome == "WICKET":
            self.wickets += 1
        elif outcome != "0":
            self.runs += int(outcome)
        
        # Update economy
        overs_decimal = self.balls / 6.0  # Use decimal for economy calculation
        self.economy = self.runs / overs_decimal if overs_decimal > 0 else 0.0

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
        # Check if innings is over due to all out
        if self.wickets >= 10:
            return ""  # No prompt needed if all out
        
        # Current over.ball for NEXT delivery (matching training format)
        # Training shows the ball about to be bowled, not current position
        next_ball_number = (self.balls % 6) + 1
        current_over = self.balls // 6
        if next_ball_number > 6:  # Start of new over
            current_over += 1
            next_ball_number = 1
        over_ball = f"{current_over}.{next_ball_number}"
        
        # 1. SITUATION - Different for 1st vs 2nd innings
        if self.innings == 2:
            # Second innings includes target info
            target = self.innings_scores.get(list(self.innings_scores.keys())[0], {}).get("runs", 0) + 1
            runs_needed = target - self.score
            balls_remaining = (20 * 6) - self.balls
            overs_remaining = balls_remaining / 6.0
            
            if overs_remaining > 0:
                rrr = round(runs_needed / overs_remaining, 1)
            else:
                rrr = 0.0
            
            situation = f"{over_ball}: {self.score}/{self.wickets} Need{runs_needed}@{rrr}"
        else:
            # First innings - simple format
            situation = f"{over_ball}: {self.score}/{self.wickets}"
        
        # 2. RECENT (last 5 balls with padding at start)
        # Get recent balls and pad to 5
        recent_list = self.recent_balls[-5:] if len(self.recent_balls) >= 5 else self.recent_balls[:]
        while len(recent_list) < 5:
            recent_list.insert(0, "-")  # Pad at beginning
        recent_str = f"Recent: {'-'.join(recent_list)}"
        
        # 3. PARTNERSHIP - shows partnership RUNS
        if self.partnership_balls > 0:
            p_rate = round((self.partnership_runs / self.partnership_balls) * 6, 1)
            partnership_str = f"P:{self.partnership_runs}@{self.partnership_balls}b({p_rate}rr)"
        else:
            partnership_str = "P:0@0b(0.0rr)"
        
        # 4. PLAYERS
        # Get last names only
        bowler_name = self.current_bowler.name.split()[-1] if " " in self.current_bowler.name else self.current_bowler.name
        striker_name = self.current_batsmen[0].name.split()[-1] if " " in self.current_batsmen[0].name else self.current_batsmen[0].name
        
        # Bowler stats
        bowler_stats = self.bowling_stats[self.bowling_team][self.current_bowler.name]
        if bowler_stats.wickets > 0:
            bowler_str = f"{bowler_name}(Econ{bowler_stats.economy:.1f},{bowler_stats.wickets}W)"
        else:
            bowler_str = f"{bowler_name}(Econ{bowler_stats.economy:.1f})"
        
        # Batsman stats - match training format exactly
        batsman_stats = self.batting_stats[self.batting_team][self.current_batsmen[0].name]
        
        if batsman_stats.balls == 0:
            # Brand new batsman
            batsman_str = f"{striker_name}(new,0 Runs @ 0 SR)"
        elif batsman_stats.balls < 10:
            # Still "new" but has faced some balls
            batsman_str = f"{striker_name}(new,{batsman_stats.runs} Runs @ {int(batsman_stats.strike_rate)} SR)"
        else:
            # Set batsman
            batsman_str = f"{striker_name}(set,{batsman_stats.runs} Runs @ {int(batsman_stats.strike_rate)} SR)"
        
        players_str = f"{bowler_str} vs {batsman_str}"
        
        # 5. CONTEXT
        # Phase with over.ball repeated
        if current_over < 6:
            phase_str = f"PP {over_ball}"
        elif current_over >= 16:
            phase_str = f"Death {over_ball}"
        else:
            # Middle overs - training data shows no phase prefix
            phase_str = None
        
        # Venue abbreviation
        venue_words = self.venue.replace(" Stadium", "").replace(" Cricket Ground", "").replace(" International", "").split()
        if len(venue_words) >= 3:
            venue_short = ''.join([w[0].upper() for w in venue_words[:3]])
        elif len(venue_words) == 2:
            venue_short = venue_words[0][:2].upper() + venue_words[1][0].upper()
        else:
            venue_short = self.venue[:3].upper()
        
        # Match string
        match_str = f"{self.batting_team} vs {self.bowling_team}, {venue_short}"
        
        # Combine context
        if phase_str:
            context = f"{phase_str} | {match_str}"
        else:
            context = match_str
        
        # Final prompt assembly
        prompt = f"{situation} | {recent_str} | {partnership_str} | {players_str} | {context}"
        
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
            
            # Get new batsman only if not all out
            if self.wickets < 10:
                new_batsman = self.get_next_batsman()
                if new_batsman:
                    self.current_batsmen[0] = new_batsman  # New batsman on strike
        else:
            runs = int(outcome)
            self.score += runs
            self.partnership_runs += runs
            
            # Rotate strike on odd runs only if not all out
            if runs % 2 == 1 and self.wickets < 10:
                self.current_batsmen[0], self.current_batsmen[1] = self.current_batsmen[1], self.current_batsmen[0]
        
        # Update partnership balls
        self.partnership_balls += 1
        
        # Update batsman stats only if not all out
        if self.wickets < 10:
            striker = self.current_batsmen[0]
            self.batting_stats[self.batting_team][striker.name].update(outcome)
        
        # Update bowler stats
        self.bowling_stats[self.bowling_team][self.current_bowler.name].update(outcome)
        
        # Record ball event
        event = BallEvent(
            over=self.overs,
            ball=(self.balls - 1) % 6 + 1,  # Ball within the over (1-6)
            bowler=self.current_bowler.name,
            batsman=self.current_batsmen[0].name if self.wickets < 10 else "ALL OUT",
            outcome=outcome,
            runs=int(outcome) if outcome != "WICKET" else 0
        )
        self.ball_by_ball.append(event)
        
        # Check for over completion
        if self.balls % 6 == 0:
            # Over is complete
            self.overs = self.balls // 6
            # Rotate strike at end of over only if not all out
            if self.wickets < 10:
                self.current_batsmen[0], self.current_batsmen[1] = self.current_batsmen[1], self.current_batsmen[0]
            # Select new bowler
            self.current_bowler = self.select_next_bowler()
    
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
    inputs = tokenizer(prompt_with_space, return_tensors="pt", max_length=96, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type=='cuda'):
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        
        # Get token IDs for outcomes
        outcome_token_ids = tokenizer.convert_tokens_to_ids(list(OUTCOME2TOK.values()))
        outcome_logits = logits[outcome_token_ids]
        
        # Use log-softmax for numerical stability
        log_probs = torch.log_softmax(outcome_logits, dim=-1)
        outcome_probs = torch.exp(log_probs).cpu().numpy()
        
        # Ensure probabilities are valid
        if np.isnan(outcome_probs).any() or np.isinf(outcome_probs).any():
            print(f"Warning: Invalid probabilities detected, using uniform distribution")
            outcome_probs = np.ones(len(OUTCOMES)) / len(OUTCOMES)
        
        # Clip and normalize
        outcome_probs = np.clip(outcome_probs, 1e-8, 1.0)
        outcome_probs = outcome_probs / outcome_probs.sum()
    
    # Create probability dictionary
    prob_dict = {outcome: prob for outcome, prob in zip(OUTCOMES, outcome_probs)}
    
    # Sample from distribution
    outcome_idx = np.random.choice(len(OUTCOMES), p=outcome_probs)
    return OUTCOMES[outcome_idx], prob_dict

def simulate_match(model, tokenizer, device, team1_name: str, team2_name: str, 
                  team1_players: List[Player], team2_players: List[Player], 
                  venue: str = "Generic Stadium", verbose: bool = False) -> Dict:
    """Simulate a complete T20 match"""
    
    match = MatchState(team1_name, team2_name, team1_players, team2_players, venue)
    
    # First innings
    match.initialize_innings()
    while not match.is_innings_complete():
        prompt = match.generate_prompt()
        outcome, prob_dict = predict_outcome(prompt, model, tokenizer, device)
        match.process_ball(outcome)
        
        if verbose and match.balls % 30 == 0:  # Print every 5 overs
            print(f"Over {match.overs}: {match.batting_team} {match.score}/{match.wickets}")
    
    # Second innings
    match.switch_innings()
    target = match.innings_scores[team1_name]["runs"] + 1
    
    while not match.is_innings_complete() and match.score < target:
        prompt = match.generate_prompt()
        outcome, prob_dict = predict_outcome(prompt, model, tokenizer, device)
        match.process_ball(outcome)
        
        if verbose and match.balls % 30 == 0:  # Print every 5 overs
            print(f"Over {match.overs}: {match.batting_team} {match.score}/{match.wickets} (Need {target - match.score})")
    
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

def load_model(model_path: str):
    """Load fine-tuned model with proper token handling"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model first
    base_model_name = "Qwen/Qwen1.5-1.8B"
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Add special tokens
    special_tokens_to_add = list(OUTCOME2TOK.values())
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,  # Manual device placement for better control
        trust_remote_code=False,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # CRITICAL: Resize embeddings to match tokenizer with special tokens
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)
    model.eval()
    
    # Compile for GPU optimization (H200 will benefit from this)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled for GPU optimization")
        except:
            print("Torch compile failed, using standard model")
    
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
    model_path = "qwen-cricket-peft-2"  # Your model path
    
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
        verbose=True  # Set to True to see progress every 5 overs
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