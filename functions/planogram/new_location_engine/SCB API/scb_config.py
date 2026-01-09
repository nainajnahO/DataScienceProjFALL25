"""
SCB API Data Fetcher Configuration
==================================

Central configuration file for fetching supermarket and company data
from the SCB (Statistics Central Bureau) API.
"""

# IMPORT DEPENDENCIES
import os
from pathlib import Path

# ==============================================================================
# API Authentication & Connection
# ==============================================================================

# Path to client certificate for mTLS authentication
CERT_PATH = 'certificate.pem'

# SCB API base URL
BASE_URL = 'https://privateapi.scb.se/nv0101/v1/sokpavar/'

# SSL verification (should always be True in production)
VERIFY_SSL = True

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# ==============================================================================
# API Endpoints
# ==============================================================================

# Work establishments (Arbetsställen) endpoints
ENDPOINT_COUNT = 'api/Ae/RaknaArbetsstallen'
ENDPOINT_FETCH = 'api/Ae/HamtaArbetsstallen'
ENDPOINT_VARIABLES = 'api/Ae/Variabler'
ENDPOINT_CATEGORIES = 'api/Ae/KategorierMedKodtabeller'

# ==============================================================================
# Industry Codes (SNI 2007)
# ==============================================================================

# Food retail codes to fetch for supermarket data
FOOD_RETAIL_CODES = [
    '47111',  # Supermarkets
    '47112',  # Små livsmedelsbutiker (Small grocery stores)
    '47119',  # Övrig livsmedelshandel (Other food retail)
    '47199',  # Kiosks / Other retail (Pressbyrån etc)
    '47241',  # Candy / Confectionery stores
    '56100'   # Restaurants / Fast Food
]

# Map SNI codes to descriptions
SNI_CODE_DESCRIPTIONS = {
    '47110': 'Butik med livsmedel',
    '47111': 'Supermarket',
    '47112': 'Små livsmedelsbutiker',
    '47119': 'Övrig livsmedelshandel',
    '47199': 'Kiosk / Övr detaljhandel',
    '47241': 'Godisbutik',
    '56100': 'Restaurang / Fast Food'
}

# Exclude establishments with SNI codes starting with these prefixes from company data
EXCLUDE_SNI_PREFIX = ['47', '56']  # Exclude all retail and restaurants from "Company" data

# ==============================================================================
# Data Filtering
# ==============================================================================

# Minimum number of employees for company data
# Recommended: 10 employees (balances dataset size with meaningful businesses)
MIN_EMPLOYEE_COUNT = 10

# Maximum results per API request (SCB API limit)
MAX_RESULTS_PER_REQUEST = 2000

# ==============================================================================
# Swedish Geographic Codes
# ==============================================================================

# County codes (Län) for pagination - all 21 Swedish counties
COUNTIES = [
    '01',  # Stockholm
    '03',  # Uppsala
    '04',  # Södermanland
    '05',  # Östergötland
    '06',  # Jönköping
    '07',  # Kronoberg
    '08',  # Kalmar
    '09',  # Gotland
    '10',  # Blekinge
    '12',  # Skåne
    '13',  # Halland
    '14',  # Västra Götaland
    '17',  # Värmland
    '18',  # Örebro
    '19',  # Västmanland
    '20',  # Dalarna
    '21',  # Gävleborg
    '22',  # Västernorrland
    '23',  # Jämtland
    '24',  # Västerbotten
    '25'   # Norrbotten
]

# County names for display (optional)
COUNTY_NAMES = {
    '01': 'Stockholm',
    '03': 'Uppsala',
    '04': 'Södermanland',
    '05': 'Östergötland',
    '06': 'Jönköping',
    '07': 'Kronoberg',
    '08': 'Kalmar',
    '09': 'Gotland',
    '10': 'Blekinge',
    '12': 'Skåne',
    '13': 'Halland',
    '14': 'Västra Götaland',
    '17': 'Värmland',
    '18': 'Örebro',
    '19': 'Västmanland',
    '20': 'Dalarna',
    '21': 'Gävleborg',
    '22': 'Västernorrland',
    '23': 'Jämtland',
    '24': 'Västerbotten',
    '25': 'Norrbotten'
}

# ==============================================================================
# Pagination Strategy
# ==============================================================================

# Employee ranges for sub-batching if a county exceeds 2000 results
EMPLOYEE_RANGES = [
    ("10", "49"),    # 10-49 employees
    ("50", "99"),    # 50-99 employees
    ("100", "199"),  # 100-199 employees
    ("200", "")      # 200+ employees (empty string means no upper limit)
]

# ==============================================================================
# Geocoding Configuration (if coordinates not provided by API)
# ==============================================================================

# Geocoding service to use: 'nominatim' (free) or 'google' (requires API key)
GEOCODING_SERVICE = 'nominatim'

# Path to geocoding cache file (JSON)
GEOCODING_CACHE_FILE = 'geocoding_cache.json'

# Delay between geocoding requests (seconds)
# Nominatim requires 1 request per second
GEOCODING_DELAY = 1.0

# Google Maps API key (if using Google geocoding)
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY', '')

# User agent for Nominatim (required)
NOMINATIM_USER_AGENT = 'zvrt_scb_data_fetcher'

# ==============================================================================
# Coordinate Validation
# ==============================================================================

# Valid Swedish coordinate ranges for validation
SWEDEN_LAT_MIN = 55.0
SWEDEN_LAT_MAX = 69.0
SWEDEN_LON_MIN = 10.0
SWEDEN_LON_MAX = 25.0

# ==============================================================================
# Output Configuration
# ==============================================================================

# Output directory (relative to SCB API folder)
OUTPUT_DIR = Path(__file__).parent.parent / 'LocationBasedForecasting' / 'data'

# Output filenames
SUPERMARKETS_FILE = 'scb_supermarkets.parquet'
COMPANIES_FILE = 'scb_companies.parquet'

# ==============================================================================
# Retry Logic
# ==============================================================================

# Maximum number of retry attempts for failed requests
MAX_RETRY_ATTEMPTS = 3

# Base delay for exponential backoff (seconds)
RETRY_BASE_DELAY = 2

# Maximum delay for exponential backoff (seconds)
RETRY_MAX_DELAY = 60

# ==============================================================================
# Logging
# ==============================================================================

# Log file format
LOG_FILE_FORMAT = 'scb_fetch_{timestamp}.log'

# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# ==============================================================================
# Testing Configuration
# ==============================================================================

# Test mode: limit to small dataset for testing
TEST_MODE = False

# Test county (Stockholm) for limited testing
TEST_COUNTY = '0180'  # Stockholm municipality

# Maximum records to fetch in test mode
TEST_MAX_RECORDS = 100

# ==============================================================================
# Field Mapping (will be populated after test_coordinates.py runs)
# ==============================================================================

# These mappings will be updated once we know the actual SCB API field names
# Populated after running test_coordinates.py to inspect response structure

# Default field mappings (may need adjustment based on actual API response)
DEFAULT_FIELD_MAPPING = {
    # Establishment/Company ID
    'id_fields': ['CFARnr', 'OrgNr', 'ForetagId', 'Id'],

    # Name fields
    'name_fields': ['Namn', 'Foretagsnamn', 'ForetagsBeteckning'],

    # Address fields
    'address_fields': ['Besöksadress', 'Besoksadress', 'Adress'],
    'city_fields': ['Besökspostort', 'Besokspostort', 'Postort'],
    'postal_code_fields': ['Postnr', 'Postnummer', 'PostNr'],

    # Coordinate fields (check for these in test_coordinates.py)
    'latitude_fields': ['Latitude', 'Lat', 'Besokslatitud', 'Y', 'Koordinat_Y'],
    'longitude_fields': ['Longitude', 'Lon', 'Long', 'Besokslongitud', 'X', 'Koordinat_X'],

    # Business classification fields
    'sni_code_fields': ['Branschkod', 'SNIKod', 'SNI'],
    'employee_count_fields': ['Anställda', 'Anstallda', 'AntalAnstallda'],

    # Geographic fields
    'county_fields': ['Län', 'LänKod', 'Lanskod'],
    'municipality_fields': ['Kommun', 'KommunKod', 'Kommunkod']
}


def get_field_value(item: dict, field_list: list, default=None):
    """
    Get value from item dict using a list of possible field names.

    Args:
        item: Dictionary containing establishment data
        field_list: List of possible field names to try
        default: Default value if no field found

    Returns:
        Value from first matching field, or default if none found
    """
    # TRY EACH FIELD NAME IN ORDER
    for field in field_list:
        if field in item:
            return item[field]
    return default


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within Swedish geographic bounds.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        True if coordinates are valid Swedish coordinates
    """
    try:
        # CONVERT TO FLOAT
        lat_f = float(lat)
        lon_f = float(lon)

        # CHECK IF WITHIN SWEDISH BOUNDS
        return (SWEDEN_LAT_MIN <= lat_f <= SWEDEN_LAT_MAX and
                SWEDEN_LON_MIN <= lon_f <= SWEDEN_LON_MAX)
    except (TypeError, ValueError):
        return False


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_absolute_cert_path() -> str:
    """Get absolute path to certificate file."""
    # RETURN AS-IS IF ALREADY ABSOLUTE
    if os.path.isabs(CERT_PATH):
        return CERT_PATH
    # MAKE RELATIVE TO THIS FILE'S DIRECTORY
    return str(Path(__file__).parent / CERT_PATH)


def get_absolute_output_dir() -> Path:
    """Get absolute path to output directory."""
    # RETURN AS-IS IF ALREADY ABSOLUTE
    if OUTPUT_DIR.is_absolute():
        return OUTPUT_DIR
    # MAKE RELATIVE TO THIS FILE'S DIRECTORY
    return (Path(__file__).parent / OUTPUT_DIR).resolve()


def ensure_output_dir_exists():
    """Create output directory if it doesn't exist."""
    # GET ABSOLUTE PATH
    output_dir = get_absolute_output_dir()
    # CREATE IF NEEDED
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == '__main__':
    # PRINT CONFIGURATION SUMMARY
    print("="*80)
    print("SCB API Data Fetcher Configuration")
    print("="*80)
    print(f"\nAPI Configuration:")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Certificate: {get_absolute_cert_path()}")
    print(f"  Verify SSL: {VERIFY_SSL}")

    print(f"\nData Filtering:")
    print(f"  Food retail SNI codes: {', '.join(FOOD_RETAIL_CODES)}")
    print(f"  Min employee count: {MIN_EMPLOYEE_COUNT}")
    print(f"  Exclude SNI prefix: {', '.join(EXCLUDE_SNI_PREFIX)}")

    print(f"\nPagination:")
    print(f"  Counties to process: {len(COUNTIES)}")
    print(f"  Max results per request: {MAX_RESULTS_PER_REQUEST}")

    print(f"\nOutput:")
    print(f"  Output directory: {get_absolute_output_dir()}")
    print(f"  Supermarkets file: {SUPERMARKETS_FILE}")
    print(f"  Companies file: {COMPANIES_FILE}")

    print(f"\nGeocoding:")
    print(f"  Service: {GEOCODING_SERVICE}")
    print(f"  Cache file: {GEOCODING_CACHE_FILE}")
    print(f"  Delay: {GEOCODING_DELAY}s per request")

    print("\n" + "="*80)
