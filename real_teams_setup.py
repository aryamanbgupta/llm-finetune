#!/usr/bin/env python3
"""
Real team setup for WI vs PAK 3rd T20I (August 4, 2025)
Based on playing XIs from the current series
"""

from simulator_v1 import Player

def create_wi_vs_pak_teams():
    """Create teams based on current WI vs PAK T20I series"""
    
    # West Indies T20I XI (based on 2nd T20I lineup)
    west_indies_players = [
        # Top Order
        Player("Alick Athanaze", "batsman", 1),      # Opening batsman
        Player("Jewel Andrew", "batsman", 2),        # Opening batsman  
        Player("Shai Hope", "batsman", 3),           # Top order batsman
        Player("Sherfane Rutherford", "batsman", 4), # Middle order
        Player("Roston Chase", "all-rounder", 5),    # Batting all-rounder
        Player("Keacy Carty", "batsman", 6),         # Middle order
        
        # Lower Order/All-rounders
        Player("Jason Holder", "all-rounder", 7),    # Premium all-rounder
        Player("Romario Shepherd", "all-rounder", 8), # All-rounder
        Player("Gudakesh Motie", "bowler", 9),       # Left-arm spinner
        Player("Shamar Joseph", "bowler", 10),       # Fast bowler
        Player("Akeal Hosein", "bowler", 11)         # Left-arm spinner
    ]
    
    # Pakistan T20I XI (based on 2nd T20I lineup)  
    pakistan_players = [
        # Top Order
        Player("Saim Ayub", "batsman", 1),           # Opening batsman
        Player("Sahibzada Farhan", "batsman", 2),    # Opening batsman
        Player("Fakhar Zaman", "batsman", 3),        # Top order
        Player("Mohammad Haris", "batsman", 4),      # Wicket-keeper batsman
        Player("Salman Agha", "all-rounder", 5),     # Captain, all-rounder
        Player("Hasan Nawaz", "batsman", 6),         # Middle order
        
        # Lower Order/All-rounders
        Player("Faheem Ashraf", "all-rounder", 7),   # All-rounder
        Player("Mohammad Nawaz", "all-rounder", 8),  # Left-arm spinner
        Player("Hasan Ali", "bowler", 9),            # Fast bowler
        Player("Shaheen Afridi", "bowler", 10),      # Left-arm fast bowler
        Player("Sufiyan Muqeem", "bowler", 11)       # Leg-spinner
    ]
    
    return west_indies_players, pakistan_players

def create_upcoming_match_config():
    """Complete configuration for the upcoming match"""
    
    wi_players, pak_players = create_wi_vs_pak_teams()
    
    return {
        "team1_name": "West Indies",
        "team2_name": "Pakistan", 
        "team1_players": wi_players,
        "team2_players": pak_players,
        "venue": "Central Broward Regional Park Stadium Turf Ground",
        "match_info": {
            "series": "Pakistan tour of West Indies 2025",
            "match": "3rd T20I",
            "date": "August 4, 2025",
            "location": "Lauderhill, USA"
        }
    }

# Usage in your parallel simulation:
if __name__ == "__main__":
    config = create_upcoming_match_config()
    
    print("=== REAL MATCH SIMULATION ===")
    print(f"Match: {config['match_info']['match']}")
    print(f"Teams: {config['team1_name']} vs {config['team2_name']}")
    print(f"Venue: {config['venue']}")
    print(f"Date: {config['match_info']['date']}")
    
    print(f"\n{config['team1_name']} XI:")
    for p in config['team1_players']:
        print(f"  {p.batting_position}. {p.name} ({p.role})")
    
    print(f"\n{config['team2_name']} XI:")
    for p in config['team2_players']:
        print(f"  {p.batting_position}. {p.name} ({p.role})")