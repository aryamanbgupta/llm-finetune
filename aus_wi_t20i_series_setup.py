#!/usr/bin/env python3
"""
Australia vs West Indies T20I Series Setup (July 2025)
5 individual match configurations from the actual series
Australia won 5-0

Series Info:
- 1st & 2nd T20I: Sabina Park, Kingston, Jamaica
- 3rd, 4th & 5th T20I: Warner Park, Basseterre, St Kitts
- Special matches: Andre Russell farewell (1st & 2nd), Owen debut (1st), 
  Kuhnemann debut (2nd), Tim David century (3rd)
"""

from simulator_v1 import Player

def create_australia_squad():
    """Australia T20I squad for the series"""
    return [
        # Top Order
        Player("Mitchell Marsh", "all-rounder", 1),      # Captain
        Player("Jake Fraser-McGurk", "batsman", 2),      # Opening batsman
        Player("Josh Inglis", "batsman", 3),             # Wicket-keeper
        Player("Cameron Green", "all-rounder", 4),       # All-rounder
        Player("Glenn Maxwell", "all-rounder", 5),       # All-rounder
        Player("Mitchell Owen", "batsman", 6),           # Middle order (debut 1st T20I)
        Player("Tim David", "batsman", 7),               # Finisher
        
        # Bowlers
        Player("Sean Abbott", "bowler", 8),              # Fast bowler
        Player("Ben Dwarshuis", "bowler", 9),            # Left-arm fast
        Player("Nathan Ellis", "bowler", 10),            # Fast bowler
        Player("Adam Zampa", "bowler", 11)               # Leg-spinner
    ]

def create_west_indies_base_squad():
    """Base West Indies squad (with variations per match)"""
    return [
        # Top Order
        Player("Shai Hope", "batsman", 1),               # Captain, wicket-keeper
        Player("Brandon King", "batsman", 2),            # Opening batsman
        Player("Jewel Andrew", "batsman", 3),            # Young batsman
        Player("Sherfane Rutherford", "batsman", 4),     # Middle order
        Player("Shimron Hetmyer", "batsman", 5),         # Middle order
        Player("Rovman Powell", "all-rounder", 6),       # All-rounder
        Player("Roston Chase", "all-rounder", 7),        # All-rounder
        
        # Bowlers
        Player("Jason Holder", "all-rounder", 8),        # Veteran all-rounder
        Player("Akeal Hosein", "bowler", 9),             # Left-arm spinner
        Player("Gudakesh Motie", "bowler", 10),          # Left-arm spinner
        Player("Alzarri Joseph", "bowler", 11)           # Fast bowler
    ]

# ============================================================================
# MATCH 1: July 20, 2025 - Sabina Park, Kingston, Jamaica
# ============================================================================
def setup_match_1():
    """1st T20I - Mitchell Owen's debut, Andre Russell farewell begins"""
    
    # Australia (Mitchell Owen debut)
    australia_xi = [
        Player("Mitchell Marsh", "all-rounder", 1),      # Captain
        Player("Jake Fraser-McGurk", "batsman", 2),      
        Player("Josh Inglis", "batsman", 3),             # Wicket-keeper
        Player("Cameron Green", "all-rounder", 4),       
        Player("Glenn Maxwell", "all-rounder", 5),       
        Player("Mitchell Owen", "batsman", 6),           # DEBUT
        Player("Cooper Connolly", "all-rounder", 7),     
        Player("Ben Dwarshuis", "bowler", 8),            
        Player("Sean Abbott", "bowler", 9),              
        Player("Nathan Ellis", "bowler", 10),            
        Player("Adam Zampa", "bowler", 11)               
    ]
    
    # West Indies (Andre Russell included)
    wi_xi = [
        Player("Shai Hope", "batsman", 1),               # Captain
        Player("Brandon King", "batsman", 2),            
        Player("Jewel Andrew", "batsman", 3),            
        Player("Sherfane Rutherford", "batsman", 4),     
        Player("Shimron Hetmyer", "batsman", 5),         
        Player("Rovman Powell", "all-rounder", 6),       
        Player("Andre Russell", "all-rounder", 7),       # FAREWELL MATCH
        Player("Jason Holder", "all-rounder", 8),        
        Player("Akeal Hosein", "bowler", 9),             
        Player("Gudakesh Motie", "bowler", 10),          
        Player("Alzarri Joseph", "bowler", 11)           
    ]
    
    return {
        "match_name": "1st T20I",
        "date": "July 20, 2025",
        "venue": "Sabina Park",
        "location": "Kingston, Jamaica",
        "team1_name": "West Indies",
        "team2_name": "Australia", 
        "team1_players": wi_xi,
        "team2_players": australia_xi,
        "special_notes": "Mitchell Owen debut, Andre Russell farewell match"
    }

# ============================================================================
# MATCH 2: July 22, 2025 - Sabina Park, Kingston, Jamaica  
# ============================================================================
def setup_match_2():
    """2nd T20I - Kuhnemann debut, Russell's final match"""
    
    # Australia (Matthew Kuhnemann debut)
    australia_xi = [
        Player("Mitchell Marsh", "all-rounder", 1),      
        Player("Jake Fraser-McGurk", "batsman", 2),      
        Player("Josh Inglis", "batsman", 3),             
        Player("Cameron Green", "all-rounder", 4),       
        Player("Glenn Maxwell", "all-rounder", 5),       
        Player("Mitchell Owen", "batsman", 6),           
        Player("Cooper Connolly", "all-rounder", 7),     
        Player("Matthew Kuhnemann", "bowler", 8),        # DEBUT (left-arm spinner)
        Player("Sean Abbott", "bowler", 9),              
        Player("Nathan Ellis", "bowler", 10),            
        Player("Adam Zampa", "bowler", 11)               
    ]
    
    # West Indies (Russell's final match)
    wi_xi = [
        Player("Shai Hope", "batsman", 1),               
        Player("Brandon King", "batsman", 2),            
        Player("Jewel Andrew", "batsman", 3),            
        Player("Sherfane Rutherford", "batsman", 4),     
        Player("Shimron Hetmyer", "batsman", 5),         
        Player("Rovman Powell", "all-rounder", 6),       
        Player("Andre Russell", "all-rounder", 7),       # FINAL MATCH
        Player("Jason Holder", "all-rounder", 8),        
        Player("Akeal Hosein", "bowler", 9),             
        Player("Gudakesh Motie", "bowler", 10),          
        Player("Alzarri Joseph", "bowler", 11)           
    ]
    
    return {
        "match_name": "2nd T20I",
        "date": "July 22, 2025", 
        "venue": "Sabina Park",
        "location": "Kingston, Jamaica",
        "team1_name": "West Indies",
        "team2_name": "Australia",
        "team1_players": wi_xi,
        "team2_players": australia_xi,
        "special_notes": "Matthew Kuhnemann debut, Andre Russell final match"
    }

# ============================================================================
# MATCH 3: July 25, 2025 - Warner Park, Basseterre, St Kitts
# ============================================================================
def setup_match_3():
    """3rd T20I - Historic centuries by David and Hope"""
    
    # Australia (Tim David in, Russell out means changes)
    australia_xi = [
        Player("Mitchell Marsh", "all-rounder", 1),      
        Player("Jake Fraser-McGurk", "batsman", 2),      
        Player("Josh Inglis", "batsman", 3),             
        Player("Cameron Green", "all-rounder", 4),       
        Player("Glenn Maxwell", "all-rounder", 5),       
        Player("Tim David", "batsman", 6),               # Record century (37 balls)
        Player("Mitchell Owen", "batsman", 7),           
        Player("Sean Abbott", "bowler", 8),              
        Player("Ben Dwarshuis", "bowler", 9),            
        Player("Nathan Ellis", "bowler", 10),            
        Player("Adam Zampa", "bowler", 11)               
    ]
    
    # West Indies (No Russell, Matthew Forde replaces)
    wi_xi = [
        Player("Shai Hope", "batsman", 1),               # Maiden T20I century
        Player("Brandon King", "batsman", 2),            
        Player("Jewel Andrew", "batsman", 3),            
        Player("Sherfane Rutherford", "batsman", 4),     
        Player("Shimron Hetmyer", "batsman", 5),         
        Player("Rovman Powell", "all-rounder", 6),       
        Player("Roston Chase", "all-rounder", 7),        
        Player("Matthew Forde", "bowler", 8),            # Replaces Russell
        Player("Akeal Hosein", "bowler", 9),             
        Player("Gudakesh Motie", "bowler", 10),          
        Player("Alzarri Joseph", "bowler", 11)           
    ]
    
    return {
        "match_name": "3rd T20I",
        "date": "July 25, 2025",
        "venue": "Warner Park", 
        "location": "Basseterre, St Kitts",
        "team1_name": "West Indies",
        "team2_name": "Australia",
        "team1_players": wi_xi,
        "team2_players": australia_xi,
        "special_notes": "Tim David fastest T20I century (37 balls), Shai Hope maiden century"
    }

# ============================================================================
# MATCH 4: July 26, 2025 - Warner Park, Basseterre, St Kitts
# ============================================================================
def setup_match_4():
    """4th T20I - Jediah Blades debut"""
    
    # Australia (similar team)
    australia_xi = [
        Player("Mitchell Marsh", "all-rounder", 1),      
        Player("Jake Fraser-McGurk", "batsman", 2),      
        Player("Josh Inglis", "batsman", 3),             
        Player("Cameron Green", "all-rounder", 4),       
        Player("Glenn Maxwell", "all-rounder", 5),       
        Player("Tim David", "batsman", 6),               
        Player("Mitchell Owen", "batsman", 7),           
        Player("Xavier Bartlett", "bowler", 8),          # Rotation option
        Player("Ben Dwarshuis", "bowler", 9),            
        Player("Nathan Ellis", "bowler", 10),            
        Player("Adam Zampa", "bowler", 11)               
    ]
    
    # West Indies (Jediah Blades debut)
    wi_xi = [
        Player("Shai Hope", "batsman", 1),               
        Player("Brandon King", "batsman", 2),            
        Player("Jewel Andrew", "batsman", 3),            
        Player("Sherfane Rutherford", "batsman", 4),     
        Player("Shimron Hetmyer", "batsman", 5),         
        Player("Rovman Powell", "all-rounder", 6),       
        Player("Roston Chase", "all-rounder", 7),        
        Player("Jason Holder", "all-rounder", 8),        
        Player("Akeal Hosein", "bowler", 9),             
        Player("Jediah Blades", "bowler", 10),           # DEBUT (fast bowler)
        Player("Gudakesh Motie", "bowler", 11)           
    ]
    
    return {
        "match_name": "4th T20I",
        "date": "July 26, 2025",
        "venue": "Warner Park",
        "location": "Basseterre, St Kitts", 
        "team1_name": "West Indies",
        "team2_name": "Australia",
        "team1_players": wi_xi,
        "team2_players": australia_xi,
        "special_notes": "Jediah Blades debut"
    }

# ============================================================================
# MATCH 5: July 28, 2025 - Warner Park, Basseterre, St Kitts
# ============================================================================
def setup_match_5():
    """5th T20I - Series finale, Evin Lewis replaced by Keacy Carty"""
    
    # Australia (similar team for finale)
    australia_xi = [
        Player("Mitchell Marsh", "all-rounder", 1),      
        Player("Jake Fraser-McGurk", "batsman", 2),      
        Player("Josh Inglis", "batsman", 3),             
        Player("Cameron Green", "all-rounder", 4),       
        Player("Glenn Maxwell", "all-rounder", 5),       
        Player("Tim David", "batsman", 6),               
        Player("Aaron Hardie", "all-rounder", 7),        # Rotation
        Player("Sean Abbott", "bowler", 8),              
        Player("Ben Dwarshuis", "bowler", 9),            
        Player("Nathan Ellis", "bowler", 10),            
        Player("Adam Zampa", "bowler", 11)               
    ]
    
    # West Indies (Keacy Carty replaces injured Evin Lewis)
    wi_xi = [
        Player("Shai Hope", "batsman", 1),               
        Player("Brandon King", "batsman", 2),            
        Player("Keacy Carty", "batsman", 3),             # Replaces Evin Lewis
        Player("Sherfane Rutherford", "batsman", 4),     
        Player("Shimron Hetmyer", "batsman", 5),         
        Player("Rovman Powell", "all-rounder", 6),       
        Player("Roston Chase", "all-rounder", 7),        
        Player("Jason Holder", "all-rounder", 8),        
        Player("Romario Shepherd", "all-rounder", 9),    # Additional all-rounder
        Player("Akeal Hosein", "bowler", 10),            
        Player("Gudakesh Motie", "bowler", 11)           
    ]
    
    return {
        "match_name": "5th T20I",
        "date": "July 28, 2025",
        "venue": "Warner Park",
        "location": "Basseterre, St Kitts",
        "team1_name": "West Indies", 
        "team2_name": "Australia",
        "team1_players": wi_xi,
        "team2_players": australia_xi,
        "special_notes": "Series finale, Keacy Carty replaces injured Evin Lewis"
    }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_all_matches():
    """Get all 5 match configurations"""
    return [
        setup_match_1(),
        setup_match_2(), 
        setup_match_3(),
        setup_match_4(),
        setup_match_5()
    ]

def get_match_by_number(match_num: int):
    """Get specific match configuration (1-5)"""
    match_functions = [
        setup_match_1,
        setup_match_2,
        setup_match_3, 
        setup_match_4,
        setup_match_5
    ]
    
    if 1 <= match_num <= 5:
        return match_functions[match_num - 1]()
    else:
        raise ValueError(f"Match number must be 1-5, got {match_num}")

def print_series_summary():
    """Print summary of all matches"""
    print("="*80)
    print("AUSTRALIA vs WEST INDIES T20I SERIES 2025 (Australia won 5-0)")
    print("="*80)
    
    matches = get_all_matches()
    for i, match in enumerate(matches, 1):
        print(f"\n{match['match_name']} - {match['date']}")
        print(f"Venue: {match['venue']}, {match['location']}")
        print(f"Special: {match['special_notes']}")
        
        print(f"\n{match['team1_name']} XI:")
        for p in match['team1_players']:
            print(f"  {p.batting_position}. {p.name} ({p.role})")
        
        print(f"\n{match['team2_name']} XI:")
        for p in match['team2_players']:
            print(f"  {p.batting_position}. {p.name} ({p.role})")
        
        if i < 5:
            print("-" * 80)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Print all match details
    print_series_summary()
    
    # Example: Get specific match
    print("\n" + "="*50)
    print("EXAMPLE: Getting Match 3 Configuration")
    print("="*50)
    
    match3 = get_match_by_number(3)
    print(f"Match: {match3['match_name']}")
    print(f"Date: {match3['date']}")
    print(f"Venue: {match3['venue']}, {match3['location']}")
    print(f"Teams: {match3['team1_name']} vs {match3['team2_name']}")
    print(f"Special: {match3['special_notes']}")

"""
USAGE IN YOUR SIMULATION:

# For a single match:
from aus_wi_t20i_series_setup import get_match_by_number

match_config = get_match_by_number(3)  # Get 3rd T20I
result = simulate_match(
    model, tokenizer, device,
    match_config['team1_name'], 
    match_config['team2_name'],
    match_config['team1_players'],
    match_config['team2_players'], 
    match_config['venue']
)

# For all 5 matches:
from aus_wi_t20i_series_setup import get_all_matches

all_matches = get_all_matches()
for match_config in all_matches:
    print(f"Simulating {match_config['match_name']}...")
    result = simulate_match(
        model, tokenizer, device,
        match_config['team1_name'],
        match_config['team2_name'], 
        match_config['team1_players'],
        match_config['team2_players'],
        match_config['venue']
    )
"""