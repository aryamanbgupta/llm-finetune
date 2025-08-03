import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

#Model setup to match training
EXTRA_BASE = 40
OUTCOMES = ["0", "1", "2", "3", "4", "6", "WICKET"]
OUTCOME2TOK = {o: f"<|extra_{EXTRA_BASE+i}|>" for i, o in enumerate(OUTCOMES)}
TOK2OUTCOME = {v: k for k, v in OUTCOME2TOK.items()}

@dataclass
class Player:
    name:str
    role:str
    batting_position:int

@dataclass
class BallEvent:
    over:int
    ball:int
    bowler: str
    batsman: str
    outcome: str
    runs: int

@dataclass
class BatsmanStats:
    runs: int =0
    balls: int =0
    fours: int = 0
    sixes: int = 0
    strike_rate: float = 0.0

    def update(self, outcome:str):
        self.balls +=1
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
    overs: float =0.0
    runs: int = 0
    wickets: int = 0
    economy: float = 0.0
    balls: int = 0

    def update(self, outcome:str):
        self.balls += 1
        self.overs = self.balls // 6 + (self.balls %6) *0.1
        if outcome == "WICKET":
            self.wickets += 1
        elif outcome != "0":
            self.runs += int(outcome)

        complete_overs = self.balls / 6
        self.economy = self.runs / complete_overs if complete_overs > 0 else 0.0

class MatchState:
    def __init__(self, team1_name: str, team2_name: str, team1_players: List[Player], team2_players: List[Player], venue: str = "PIC"):
        self.team1_name = team1_name
        self.team2_name = team2_name
        self.team1_players = team1_players
        self.team2_players = team2_players
        self.venue = venue

        self.innings = 1
        self.batting_team = team1_name
        self.bowling_team = team2_name
        self.current_batsmen = []
        self.current_bowler = None
        self.score = 0
        self.wickets = 0
        self.overs = 0
        self.balls =0

        self.recent_balls = []  # Last 5 balls
        self.ball_by_ball = []

        self.batting_stats = {team1_name: defaultdict(BatsmanStats), team2_name: defaultdict(BowlerStats)}
        self.bowling_stats = {team1_name: defaultdict(BowlerStats), team2_name: defaultdict(BatsmanStats)}

        self.partnership_runs = 0
        self.partnership_balls = 0

        self.innings_scores = {}