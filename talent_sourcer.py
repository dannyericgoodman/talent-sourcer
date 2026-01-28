#!/usr/bin/env python3
"""
Superior Studios Talent Sourcer
================================
Automated tool to identify high-potential tech talent in the Illinois/Chicago area.
Searches for senior leaders, serial founders, and pedigree talent, then scores them
using an LLM and outputs results to Google Sheets.

Author: Built for Superior Studios
Version: 1.0.0
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional
import time

import requests
from serpapi import GoogleSearch
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys (set via environment variables for security)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Google Sheets Configuration
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")  # The ID from your sheet URL
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")  # Service account JSON

# Search Configuration
LOCATION = "Illinois, United States"
LOCATION_KEYWORDS = ["Chicago", "Illinois", "IL", "Chicagoland"]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SEARCH QUERIES - Customize these to refine your talent search
# ============================================================================

SEARCH_QUERIES = [
    # ==========================================================================
    # STUDENT & EARLY-CAREER BUILDERS (Pre-YC Targets)
    # ==========================================================================

    # University of Chicago - CS/Engineering Students & Recent Grads
    'site:linkedin.com/in "University of Chicago" "Computer Science" founder OR building OR startup',
    'site:linkedin.com/in "UChicago" software engineer intern OR "side project" OR hackathon',
    'site:linkedin.com/in "University of Chicago" "Machine Learning" OR "AI" student',
    'site:linkedin.com/in "Booth School" MBA entrepreneur OR founder OR venture',

    # Northwestern - CS/Engineering Students & Recent Grads
    'site:linkedin.com/in "Northwestern University" "Computer Science" founder OR startup OR building',
    'site:linkedin.com/in "Northwestern" software engineer "side project" OR hackathon winner',
    'site:linkedin.com/in "Kellogg" MBA entrepreneur OR founder Chicago',
    'site:linkedin.com/in "Northwestern University" "Machine Learning" OR "AI" student OR researcher',

    # University of Illinois (UIUC & UIC)
    'site:linkedin.com/in "University of Illinois" "Computer Science" founder OR startup Chicago',
    'site:linkedin.com/in "UIUC" software engineer Chicago founder OR building',
    'site:linkedin.com/in "Grainger College" engineering startup OR founder',
    'site:linkedin.com/in "UIC" computer science entrepreneur OR founder Chicago',

    # Illinois Institute of Technology
    'site:linkedin.com/in "Illinois Tech" OR "IIT" computer science founder OR startup Chicago',
    'site:linkedin.com/in "Illinois Institute of Technology" engineering entrepreneur',

    # DePaul & Loyola
    'site:linkedin.com/in "DePaul University" computer science startup OR founder Chicago',
    'site:linkedin.com/in "Loyola University Chicago" software engineer founder OR entrepreneur',

    # Student Hackathon Winners & Builders
    'site:linkedin.com/in Chicago "hackathon winner" OR "hackathon first place" student',
    'site:linkedin.com/in Chicago student "built" OR "building" startup OR "side project"',
    'site:linkedin.com/in Illinois "HackIllinois" OR "WildHacks" OR "MHacks" winner',
    'site:linkedin.com/in Chicago "YC applicant" OR "applying to YC" student',

    # Thiel Fellowship & Other Student Entrepreneur Programs
    'site:linkedin.com/in "Thiel Fellow" OR "Thiel Fellowship" Chicago OR Illinois',
    'site:linkedin.com/in "Contrary" OR "Contrary Capital" Chicago student OR fellow',
    'site:linkedin.com/in "Neo" scholar OR fellow Chicago Illinois',
    'site:linkedin.com/in "Rough Draft Ventures" Chicago OR Illinois',
    'site:linkedin.com/in "Dorm Room Fund" Chicago OR Illinois',

    # University Incubators & Entrepreneurship Programs
    'site:linkedin.com/in "Polsky Center" Chicago entrepreneur OR founder',
    'site:linkedin.com/in "The Garage" Northwestern startup OR founder',
    'site:linkedin.com/in "1871" Chicago student OR intern founder',
    'site:linkedin.com/in "MATTER" Chicago healthtech founder student',
    'site:linkedin.com/in "mHub" Chicago hardware founder student',

    # Recent Grads at Top Tech (Potential Boomerangs)
    'site:linkedin.com/in Chicago "new grad" OR "recent graduate" Google OR Meta OR Apple software',
    'site:linkedin.com/in Illinois "class of 2024" OR "class of 2023" software engineer startup',
    'site:linkedin.com/in Chicago junior engineer "side project" OR "building" OR "founded"',

    # Graduate Students & PhD Candidates (Deep Tech Founders)
    'site:linkedin.com/in Chicago "PhD" OR "Ph.D." "Machine Learning" OR "AI" startup OR founder',
    'site:linkedin.com/in "University of Chicago" "PhD candidate" computer science',
    'site:linkedin.com/in "Northwestern" "PhD" artificial intelligence OR robotics Chicago',
    'site:linkedin.com/in Illinois "postdoc" OR "post-doc" AI OR ML founder OR startup',

    # Young Technical PMs (Future Founder Archetype)
    'site:linkedin.com/in Chicago "Associate Product Manager" OR "APM" Google OR Meta OR Uber',
    'site:linkedin.com/in Illinois "Product Manager" "new grad" OR junior tech startup',

    # ==========================================================================
    # SENIOR ENGINEERING/PRODUCT LEADERS
    # ==========================================================================
    'site:linkedin.com/in "VP Engineering" OR "VP of Engineering" Chicago Illinois',
    'site:linkedin.com/in "Head of Engineering" Chicago Illinois',
    'site:linkedin.com/in "Director of Engineering" Chicago Illinois',
    'site:linkedin.com/in "VP Product" OR "VP of Product" Chicago Illinois',
    'site:linkedin.com/in "Head of Product" Chicago Illinois',
    'site:linkedin.com/in "CTO" Chicago Illinois startup',
    'site:linkedin.com/in "Chief Technology Officer" Chicago Illinois',
    'site:linkedin.com/in "Principal Engineer" Chicago Illinois',
    'site:linkedin.com/in "Staff Engineer" Chicago Illinois',

    # ==========================================================================
    # SERIAL FOUNDERS
    # ==========================================================================
    'site:linkedin.com/in "Founder" "exited" OR "acquired" Chicago Illinois',
    'site:linkedin.com/in "Co-Founder" "sold" OR "exit" Chicago Illinois',
    'site:linkedin.com/in "Serial Entrepreneur" Chicago Illinois',
    'site:linkedin.com/in "Former Founder" Chicago Illinois',
    'site:linkedin.com/in "Entrepreneur in Residence" OR "EIR" Chicago Illinois',

    # ==========================================================================
    # YC AND ACCELERATOR ALUMNI
    # ==========================================================================
    'site:linkedin.com/in "Y Combinator" OR "YC" Chicago Illinois',
    'site:linkedin.com/in "Techstars" Chicago Illinois founder',
    'site:linkedin.com/in "500 Startups" Chicago Illinois',
    'site:linkedin.com/in "a16z" Chicago Illinois',

    # ==========================================================================
    # HYPERSCALE COMPANY ALUMNI
    # ==========================================================================
    'site:linkedin.com/in "OpenAI" Chicago Illinois',
    'site:linkedin.com/in "Anthropic" Chicago Illinois',
    'site:linkedin.com/in "Stripe" Chicago Illinois engineer OR product',
    'site:linkedin.com/in "Airbnb" Chicago Illinois engineer OR product',
    'site:linkedin.com/in "Uber" Chicago Illinois engineer OR product manager',
    'site:linkedin.com/in "Coinbase" Chicago Illinois',
    'site:linkedin.com/in "Square" OR "Block" Chicago Illinois engineer',
    'site:linkedin.com/in "Palantir" Chicago Illinois',
    'site:linkedin.com/in "SpaceX" Chicago Illinois',
    'site:linkedin.com/in "Scale AI" Chicago Illinois',
    'site:linkedin.com/in "Databricks" Chicago Illinois',
    'site:linkedin.com/in "Snowflake" Chicago Illinois',

    # ==========================================================================
    # CHICAGO-SPECIFIC TECH HUBS
    # ==========================================================================
    'site:linkedin.com/in "Grubhub" Chicago founder OR VP OR director',
    'site:linkedin.com/in "Groupon" Chicago founder OR VP OR engineering',
    'site:linkedin.com/in "Avant" Chicago engineering OR product',
    'site:linkedin.com/in "Tempus" Chicago engineering OR product',
    'site:linkedin.com/in "Sprout Social" Chicago VP OR director OR founder',
]

# ============================================================================
# LLM SCORING PROMPT
# ============================================================================

SCORING_PROMPT = """You are an expert talent scout for a Chicago-based venture studio looking to identify potential startup founders BEFORE they get into YC or other accelerators.

We are looking for both experienced operators AND early-career builders (students, recent grads) with exceptional potential.

Analyze the following profile and score their "Founder Potential" on a scale of 1-10.

SCORING CRITERIA:

EXPERIENCED TALENT:
- 9-10: Serial founder with exits, C-level at hypergrowth startup, or deep expertise at elite company (OpenAI, Stripe, etc.)
- 7-8: VP/Director at notable tech company, first-time founder with traction, early employee at successful startup
- 5-6: Senior IC or manager at good tech company with entrepreneurial signals

STUDENT & EARLY-CAREER TALENT (score them on POTENTIAL, not experience):
- 9-10: Exceptional student builder - already launched products/companies, hackathon winners, Thiel Fellow, has shipped real software used by others
- 7-8: Strong student builder - active side projects, technical blog/portfolio, leadership in tech clubs, interned at top companies, building something now
- 5-6: Promising student - CS/Engineering at top school, some projects, shows entrepreneurial curiosity
- 3-4: Standard student - good school but no builder signals
- 1-2: Not a fit - non-technical, no entrepreneurial indicators

KEY SIGNALS TO LOOK FOR IN STUDENTS:
- "Building" or "Built" anything in their headline
- Hackathon wins or participation
- Side projects or personal apps
- Technical internships at startups or FAANG
- Leadership in entrepreneurship/tech clubs
- PhD/research in AI/ML (deep tech founder potential)
- Thiel Fellow, Contrary, Neo, Dorm Room Fund

PROFILE TO ANALYZE:
Name: {name}
Title/Headline: {title}
Snippet/Bio: {snippet}
Source URL: {url}

Respond with ONLY a JSON object in this exact format:
{{
    "score": <number 1-10>,
    "archetype": "<one of: 'Student Builder', 'PhD/Researcher', 'Recent Grad', 'Senior Leader', 'Serial Founder', 'Pedigree Talent', 'Promising', 'Not a Fit'>",
    "reasoning": "<2-3 sentence explanation>",
    "key_signals": ["<signal 1>", "<signal 2>", "<signal 3>"]
}}
"""

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def generate_profile_id(url: str, name: str) -> str:
    """Generate a unique ID for a profile to handle deduplication."""
    unique_string = f"{url}_{name}".lower().strip()
    return hashlib.md5(unique_string.encode()).hexdigest()[:12]


def search_talent(query: str, num_results: int = 10) -> list[dict]:
    """
    Search for talent profiles using SerpApi.
    Returns a list of profile dictionaries.
    """
    if not SERPAPI_KEY:
        logger.error("SERPAPI_KEY not set!")
        return []

    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "gl": "us",  # Google country
            "hl": "en",  # Language
        })

        results = search.get_dict()
        profiles = []

        for result in results.get("organic_results", []):
            # Extract profile info from search results
            profile = {
                "name": extract_name_from_title(result.get("title", "")),
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("link", ""),
                "source": "linkedin" if "linkedin.com" in result.get("link", "") else "other",
            }

            # Only include LinkedIn profiles
            if profile["source"] == "linkedin" and profile["name"]:
                profile["profile_id"] = generate_profile_id(profile["url"], profile["name"])
                profiles.append(profile)

        logger.info(f"Found {len(profiles)} profiles for query: {query[:50]}...")
        return profiles

    except Exception as e:
        logger.error(f"Search error for query '{query[:50]}...': {e}")
        return []


def extract_name_from_title(title: str) -> str:
    """Extract person's name from LinkedIn title format."""
    # LinkedIn titles are usually "Name - Title | LinkedIn" or "Name | LinkedIn"
    if not title:
        return ""

    # Remove " | LinkedIn" suffix
    name_part = title.replace(" | LinkedIn", "").replace(" - LinkedIn", "")

    # Take the part before the first " - " (usually the name)
    if " - " in name_part:
        name_part = name_part.split(" - ")[0]

    return name_part.strip()


def score_profile_with_llm(profile: dict) -> dict:
    """
    Use OpenAI to score a profile's founder potential.
    Returns the profile with added scoring fields.
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set!")
        return {**profile, "score": 0, "archetype": "Error", "reasoning": "API key not configured"}

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = SCORING_PROMPT.format(
            name=profile.get("name", "Unknown"),
            title=profile.get("title", "Unknown"),
            snippet=profile.get("snippet", "No additional info"),
            url=profile.get("url", "")
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective and fast
            messages=[
                {"role": "system", "content": "You are a talent scout. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for consistent scoring
            max_tokens=500
        )

        # Parse the LLM response
        response_text = response.choices[0].message.content.strip()

        # Clean up response if needed (remove markdown code blocks)
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        scoring = json.loads(response_text)

        return {
            **profile,
            "score": scoring.get("score", 0),
            "archetype": scoring.get("archetype", "Unknown"),
            "reasoning": scoring.get("reasoning", ""),
            "key_signals": ", ".join(scoring.get("key_signals", [])),
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response for {profile.get('name')}: {e}")
        return {**profile, "score": 5, "archetype": "Parse Error", "reasoning": "Could not parse LLM response"}
    except Exception as e:
        logger.error(f"LLM scoring error for {profile.get('name')}: {e}")
        return {**profile, "score": 0, "archetype": "Error", "reasoning": str(e)}


def get_google_sheets_client():
    """Initialize and return Google Sheets client."""
    if not GOOGLE_CREDENTIALS_JSON:
        raise ValueError("GOOGLE_CREDENTIALS_JSON not set!")

    # Parse credentials from environment variable
    creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(credentials)

    return client


def get_existing_profile_ids(worksheet) -> set:
    """Get all existing profile IDs from the spreadsheet to avoid duplicates."""
    try:
        # Get all values from column A (profile_id column)
        all_values = worksheet.col_values(1)
        # Skip header row
        return set(all_values[1:]) if len(all_values) > 1 else set()
    except Exception as e:
        logger.error(f"Error getting existing IDs: {e}")
        return set()


def setup_spreadsheet(client, sheet_id: str):
    """Ensure the spreadsheet has the correct headers."""
    spreadsheet = client.open_by_key(sheet_id)

    try:
        worksheet = spreadsheet.worksheet("Talent Pipeline")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title="Talent Pipeline", rows=1000, cols=15)

    # Check if headers exist
    try:
        first_row = worksheet.row_values(1)
    except:
        first_row = []

    headers = [
        "Profile ID",
        "Name",
        "Title/Headline",
        "Score (1-10)",
        "Archetype",
        "Key Signals",
        "Reasoning",
        "LinkedIn URL",
        "Date Added",
        "Status",
        "Notes"
    ]

    if not first_row or first_row[0] != "Profile ID":
        worksheet.update('A1:K1', [headers])
        # Format header row
        worksheet.format('A1:K1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.6}
        })
        logger.info("Spreadsheet headers created")

    return worksheet


def add_profiles_to_sheet(worksheet, profiles: list[dict], existing_ids: set) -> int:
    """Add new profiles to the Google Sheet. Returns count of new additions."""
    new_count = 0
    rows_to_add = []

    for profile in profiles:
        profile_id = profile.get("profile_id", "")

        # Skip if already exists
        if profile_id in existing_ids:
            logger.debug(f"Skipping duplicate: {profile.get('name')}")
            continue

        # Skip low-scoring profiles (below 4)
        if profile.get("score", 0) < 4:
            logger.debug(f"Skipping low-score profile: {profile.get('name')} (score: {profile.get('score')})")
            continue

        row = [
            profile_id,
            profile.get("name", ""),
            profile.get("title", ""),
            profile.get("score", 0),
            profile.get("archetype", ""),
            profile.get("key_signals", ""),
            profile.get("reasoning", ""),
            profile.get("url", ""),
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            "New",  # Status
            ""  # Notes (empty for user to fill)
        ]

        rows_to_add.append(row)
        existing_ids.add(profile_id)  # Prevent duplicates within this run
        new_count += 1

    # Batch add rows for efficiency
    if rows_to_add:
        # Find the next empty row
        all_values = worksheet.get_all_values()
        next_row = len(all_values) + 1

        # Add all new rows at once
        cell_range = f'A{next_row}:K{next_row + len(rows_to_add) - 1}'
        worksheet.update(cell_range, rows_to_add)
        logger.info(f"Added {new_count} new profiles to spreadsheet")

    return new_count


def run_talent_sourcing():
    """Main function to run the complete talent sourcing pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Superior Studios Talent Sourcing Run")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Validate configuration
    missing_config = []
    if not SERPAPI_KEY:
        missing_config.append("SERPAPI_KEY")
    if not OPENAI_API_KEY:
        missing_config.append("OPENAI_API_KEY")
    if not GOOGLE_SHEET_ID:
        missing_config.append("GOOGLE_SHEET_ID")
    if not GOOGLE_CREDENTIALS_JSON:
        missing_config.append("GOOGLE_CREDENTIALS_JSON")

    if missing_config:
        logger.error(f"Missing required configuration: {', '.join(missing_config)}")
        return

    # Initialize Google Sheets
    try:
        sheets_client = get_google_sheets_client()
        worksheet = setup_spreadsheet(sheets_client, GOOGLE_SHEET_ID)
        existing_ids = get_existing_profile_ids(worksheet)
        logger.info(f"Connected to Google Sheets. Existing profiles: {len(existing_ids)}")
    except Exception as e:
        logger.error(f"Failed to connect to Google Sheets: {e}")
        return

    # Run searches and collect profiles
    all_profiles = []

    for i, query in enumerate(SEARCH_QUERIES):
        logger.info(f"Running search {i+1}/{len(SEARCH_QUERIES)}")

        profiles = search_talent(query, num_results=10)

        # Filter out duplicates before scoring (to save API calls)
        new_profiles = [p for p in profiles if p.get("profile_id") not in existing_ids]

        # Score each new profile
        for profile in new_profiles:
            if profile.get("profile_id") not in existing_ids:
                scored_profile = score_profile_with_llm(profile)
                all_profiles.append(scored_profile)

                # Small delay to avoid rate limiting
                time.sleep(0.5)

        # Rate limiting for SerpApi (be respectful)
        time.sleep(2)

    # Remove duplicates from this run
    seen_ids = set()
    unique_profiles = []
    for p in all_profiles:
        if p.get("profile_id") not in seen_ids:
            seen_ids.add(p.get("profile_id"))
            unique_profiles.append(p)

    logger.info(f"Total unique profiles found: {len(unique_profiles)}")

    # Sort by score (highest first)
    unique_profiles.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Add to spreadsheet
    new_count = add_profiles_to_sheet(worksheet, unique_profiles, existing_ids)

    # Summary
    logger.info("=" * 60)
    logger.info("SOURCING RUN COMPLETE")
    logger.info(f"New profiles added: {new_count}")
    logger.info(f"Total profiles in pipeline: {len(existing_ids) + new_count}")
    logger.info("=" * 60)

    # Print top finds for this run
    top_profiles = [p for p in unique_profiles if p.get("score", 0) >= 7][:5]
    if top_profiles:
        logger.info("\nTOP FINDS THIS RUN:")
        for p in top_profiles:
            logger.info(f"  [{p.get('score')}] {p.get('name')} - {p.get('archetype')}")


if __name__ == "__main__":
    run_talent_sourcing()
