"""Reusable utilities for NBA injury data collection, cleaning, and integration."""
import time
import pandas as pd
from typing import List, Optional


# =============================================================================
# PLAYER NAME MAPPINGS
# =============================================================================

# Manual mappings for players with name variations between data sources
MANUAL_NAME_MAPPINGS = {
    # Special characters (elap733 -> nba_api)
    # Note: NBA API sometimes uses special chars, sometimes doesn't
    "Luka Doncic": "Luka Dončić",
    "Dario Saric": "Dario Šarić",
    "Bojan Bogdanovic": "Bojan Bogdanović",
    "Bogdan Bogdanovic": "Bogdan Bogdanović",
    "Nikola Jokic": "Nikola Jokić",
    "Nikola Vucevic": "Nikola Vučević",
    "Jonas Valanciunas": "Jonas Valančiūnas",
    "Jusuf Nurkic": "Jusuf Nurkić",
    "Ante Zizic": "Ante Zizic",  # NBA API has no special chars
    "Goran Dragic": "Goran Dragić",
    "Kristaps Porzingis": "Kristaps Porziņģis",
    "Davis Bertans": "Davis Bertans",  # NBA API has no special chars
    "Donatas Motiejunas": "Donatas Motiejūnas",
    "Timofey Mozgov": "Timofey Mozgov",
    "Ersan Ilyasova": "Ersan İlyasova",

    # Suffix mismatches
    "Mike Conley Jr.": "Mike Conley",
    "Glen Rice Jr.": "Glen Rice",
    "Mike Dunleavy Jr.": "Mike Dunleavy",
    "Larry Nance Jr.": "Larry Nance Jr.",
    "Tim Hardaway Jr.": "Tim Hardaway Jr.",
    "Gary Payton II": "Gary Payton II",
    "Glenn Robinson III": "Glenn Robinson III",
    "Harry Giles": "Harry Giles III",
    "Frank Mason": "Frank Mason III",
    "Tony Wroten Jr.": "Tony Wroten",
    "Wayne Selden Jr.": "Wayne Selden",

    # Legal name vs nickname
    "Edrice Adebayo": "Bam Adebayo",
    "Ogugua Anunoby": "OG Anunoby",
    "Patrick Mills": "Patty Mills",
    "Sviatoslav Mykhailiuk": "Svi Mykhailiuk",
    "Guillermo Hernangomez": "Willy Hernangomez",
    "Maximilian Kleber": "Maxi Kleber",
    "Nene Hilario": "Nene",
    "Luigi Datome": "Gigi Datome",
    "Ishmael Smith": "Ish Smith",
    "Maurice Williams": "Mo Williams",
    "James Ennis": "James Ennis III",
    "Jose Juan Barea": "J.j. Barea",
    "Mohamed Bamba": "Mo Bamba",

    # Other name variations
    "Hidayet Turkoglu": "Hedo Turkoglu",
    "Predrag Stojakovic": "Peja Stojakovic",
    "Zaur Pachulia": "Zaza Pachulia",
    "(william) Tony Parker": "Tony Parker",
    "Raulzinho Neto": "Raul Neto",
    "Luc Richard Mbah A Moute": "Luc Mbah A Moute",
    "Kahlil Felder": "Kay Felder",
    "Enes Kanter": "Enes Freedom",  # Legal name change

    # Metta World Peace is his actual name in the data
    "Metta World Peace": "Metta World Peace",
}

# Coaches/executives to exclude from injury data
# These appear in transaction data but aren't players
EXCLUDE_FROM_INJURY_DATA = [
    "Steve Clifford",
    "Tyronn Lue",
    "Steve Kerr",
    "Kevin Mchale",
    "Kevin McHale",
    "Danny Ferry",
    "Andy Roeser",
    "Gregg Popovich",
    "Doc Rivers",
    "Mike Budenholzer",
    "Erik Spoelstra",
    "Rick Carlisle",
    "Bill Walker",  # Scout, not player
    "John Collins (martin)",  # Data error
    "John Wallace",  # Last played 2003, false fuzzy match to John Wall
    "Monty Williams",  # Coach, false fuzzy match to Mo Williams
]


# =============================================================================
# NBA API RATE LIMITING
# =============================================================================

# NBA API rate limiting
def rate_limited_call(func, delay: float = 1.0, *args, **kwargs):
    """
    Execute a function with rate limiting.
    
    Parameters
    ----------
    func : callable
        Function to call
    delay : float
        Seconds to wait before call
    """
    time.sleep(delay)
    return func(*args, **kwargs)


def get_season_dates(season: str) -> dict:
    """
    Get start and end dates for an NBA season.
    
    Parameters
    ----------
    season : str
        Season in 'YYYY-YY' format (e.g., '2023-24')
    
    Returns
    -------
    dict
        Dictionary with 'start' and 'end' date strings
    """
    seasons = {
        '2018-19': {'start': '2018-10-01', 'end': '2019-06-30'},
        '2019-20': {'start': '2019-10-01', 'end': '2020-10-15'},
        '2020-21': {'start': '2020-12-01', 'end': '2021-07-31'},
        '2021-22': {'start': '2021-10-01', 'end': '2022-06-30'},
        '2022-23': {'start': '2022-10-01', 'end': '2023-06-30'},
        '2023-24': {'start': '2023-10-01', 'end': '2024-06-30'},
        '2024-25': {'start': '2024-10-01', 'end': '2025-06-30'},
    }
    return seasons.get(season, {'start': None, 'end': None})


def parse_player_name(name: str) -> dict:
    """
    Parse a player name string into components.
    
    Parameters
    ----------
    name : str
        Player name (e.g., "LeBron James" or "James, LeBron")
    
    Returns
    -------
    dict
        Dictionary with 'first_name', 'last_name', 'full_name'
    """
    if not name or pd.isna(name):
        return {'first_name': '', 'last_name': '', 'full_name': ''}
    
    name = str(name).strip()
    
    if ',' in name:
        parts = name.split(',')
        last_name = parts[0].strip()
        first_name = parts[1].strip() if len(parts) > 1 else ''
    else:
        parts = name.split()
        first_name = parts[0] if parts else ''
        last_name = ' '.join(parts[1:]) if len(parts) > 1 else ''
    
    return {
        'first_name': first_name,
        'last_name': last_name,
        'full_name': f"{first_name} {last_name}".strip()
    }


def calculate_back_to_back_games(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate back-to-back game occurrences for each team.
    
    Parameters
    ----------
    df_games : pd.DataFrame
        DataFrame with columns: TEAM_ID, GAME_DATE
    
    Returns
    -------
    pd.DataFrame
        DataFrame with back-to-back flags added
    """
    df = df_games.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['TEAM_ID', 'GAME_DATE'])
    
    # Calculate days since last game for each team
    df['days_since_last'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
    
    # Flag back-to-back games (played day after previous game)
    df['is_back_to_back'] = df['days_since_last'] == 1
    
    return df


def assign_season(date: pd.Timestamp) -> Optional[str]:
    """
    Assign an NBA season to a date.
    
    Parameters
    ----------
    date : pd.Timestamp
        Date to assign
    
    Returns
    -------
    str or None
        Season string (e.g., '2023-24') or None if outside known seasons
    """
    if pd.isna(date):
        return None
    
    year = date.year
    month = date.month
    
    # NBA season runs roughly Oct-June
    if month >= 10:  # Oct-Dec: first year of season
        return f"{year}-{str(year + 1)[-2:]}"
    elif month <= 6:  # Jan-June: second year of season
        return f"{year - 1}-{str(year)[-2:]}"
    else:  # July-Sept: offseason
        return None


def standardize_team_name(team: str) -> str:
    """
    Standardize team name variations.
    
    Parameters
    ----------
    team : str
        Team name or abbreviation
    
    Returns
    -------
    str
        Standardized team name
    """
    team_mapping = {
        'PHX': 'Phoenix Suns',
        'PHO': 'Phoenix Suns',
        'BKN': 'Brooklyn Nets',
        'BRK': 'Brooklyn Nets',
        'NJN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets',
        'CHO': 'Charlotte Hornets',
        'CHH': 'Charlotte Hornets',
        'NOH': 'New Orleans Pelicans',
        'NOP': 'New Orleans Pelicans',
        'NOK': 'New Orleans Pelicans',
        'WSH': 'Washington Wizards',
        'WAS': 'Washington Wizards',
        'GS': 'Golden State Warriors',
        'GSW': 'Golden State Warriors',
        'SA': 'San Antonio Spurs',
        'SAS': 'San Antonio Spurs',
        'NY': 'New York Knicks',
        'NYK': 'New York Knicks',
        'LA': 'Los Angeles Lakers',
        'LAL': 'Los Angeles Lakers',
        'LAC': 'Los Angeles Clippers',
        'OKC': 'Oklahoma City Thunder',
    }
    
    if not team:
        return team
    
    team_upper = str(team).upper().strip()
    return team_mapping.get(team_upper, team)
