"""
SCB API Data Fetcher

Fetches supermarket and company data from SCB (Statistics Central Bureau) API
and saves to parquet files for LocationBasedForecasting system.

Usage:
    python scb_data_fetcher.py

Output:
    - LocationBasedForecasting/data/scb_supermarkets.parquet
    - LocationBasedForecasting/data/scb_companies.parquet
"""

# IMPORT DEPENDENCIES
import requests
import pandas as pd
import numpy as np
from pyproj import Transformer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import logging
from datetime import datetime

# IMPORT CONFIGURATION
import scb_config as config

# Setup Logging
# --------------------------------

# CREATE LOG FILE WITH TIMESTAMP
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f'scb_fetch_{timestamp}.log'

# CONFIGURE LOGGING
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# Coordinate Conversion
# --------------------------------

# INITIALIZE COORDINATE TRANSFORMER (SWEREF99 → WGS84)
transformer = Transformer.from_crs("EPSG:3006", "EPSG:4326", always_xy=True)


def sweref99_to_wgs84(north: float, east: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert SWEREF99 coordinates to WGS84 (lat/lon).

    Args:
        north: North coordinate in SWEREF99 (meters)
        east: East coordinate in SWEREF99 (meters)

    Returns:
        Tuple of (latitude, longitude) or (None, None) if conversion fails
    """
    try:
        # CONVERT COORDINATES (SWEREF99 TM USES EAST, NORTH ORDER)
        lon, lat = transformer.transform(east, north)

        # VALIDATE COORDINATES ARE WITHIN SWEDISH BOUNDS
        if config.validate_coordinates(lat, lon):
            return lat, lon
        else:
            logger.warning(f"Converted coordinates out of range: lat={lat}, lon={lon}")
            return None, None

    except Exception as e:
        logger.error(f"Coordinate conversion failed for north={north}, east={east}: {e}")
        return None, None



# API Client
# --------------------------------

class SCBAPIClient:
    """
    Client for SCB API communication with mTLS authentication.
    """

    def __init__(self, cert_path: str, base_url: str = config.BASE_URL):
        """
        Initialize SCB API client.

        Args:
            cert_path: Path to client certificate for mTLS
            base_url: Base URL for SCB API
        """
        self.cert_path = config.get_absolute_cert_path()
        self.base_url = base_url
        self.session = requests.Session()

        logger.info(f"Initialized SCB API Client")
        logger.info(f"  Certificate: {self.cert_path}")
        logger.info(f"  Base URL: {self.base_url}")

    @retry(
        stop=stop_after_attempt(config.MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=config.RETRY_BASE_DELAY, max=config.RETRY_MAX_DELAY),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
    )
    def _make_request(self, endpoint: str, payload: Dict, method: str = 'POST') -> Dict:
        """
        Make HTTP request to SCB API with retry logic.

        Args:
            endpoint: API endpoint (relative to base URL)
            payload: Request payload (for POST)
            method: HTTP method ('GET' or 'POST')

        Returns:
            Response data as dictionary

        Raises:
            requests.exceptions.HTTPError: On HTTP errors
            requests.exceptions.SSLError: On SSL/certificate errors
        """
        # BUILD FULL URL
        url = f"{self.base_url}{endpoint}"

        try:
            # MAKE HTTP REQUEST
            if method == 'GET':
                response = self.session.get(
                    url,
                    cert=self.cert_path,
                    verify=config.VERIFY_SSL,
                    timeout=config.REQUEST_TIMEOUT
                )
            else:  # POST
                response = self.session.post(
                    url,
                    json=payload,
                    cert=self.cert_path,
                    verify=config.VERIFY_SSL,
                    timeout=config.REQUEST_TIMEOUT
                )

            # HANDLE RATE LIMITING
            if response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting 60s...")
                time.sleep(60)
                raise requests.exceptions.ConnectionError("Rate limited")

            # CHECK FOR HTTP ERRORS
            response.raise_for_status()

            # RETURN JSON RESPONSE
            return response.json()

        except requests.exceptions.SSLError as e:
            logger.error(f"SSL Error: {e}. Check certificate path and permissions.")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e.response.text[:500]}")
            raise

    def count_establishments(self, filters: Dict) -> int:
        """
        Count establishments matching filters.

        Args:
            filters: Filter payload

        Returns:
            Number of matching establishments
        """
        try:
            # MAKE COUNT REQUEST
            result = self._make_request(config.ENDPOINT_COUNT, filters)

            # PARSE COUNT FROM RESPONSE
            if isinstance(result, int):
                return result
            elif isinstance(result, dict):
                count = result.get('Count', result.get('count', 0))
                return int(count)
            else:
                logger.warning(f"Unexpected count result type: {type(result)}")
                return 0
        except Exception as e:
            logger.error(f"Failed to count establishments: {e}")
            return 0

    def fetch_establishments(self, filters: Dict) -> List[Dict]:
        """
        Fetch establishments matching filters.

        Args:
            filters: Filter payload

        Returns:
            List of establishment dictionaries
        """
        try:
            # MAKE FETCH REQUEST
            result = self._make_request(config.ENDPOINT_FETCH, filters)

            # VALIDATE RESPONSE TYPE
            if isinstance(result, list):
                return result
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                return []

        except Exception as e:
            logger.error(f"Failed to fetch establishments: {e}")
            return []
    def get_municipalities(self, county_code: str) -> List[str]:
        """
        Get all municipality codes for a specific county.
        
        Args:
            county_code: Two-digit county code (e.g., '01')
            
        Returns:
            List of four-digit municipality codes (e.g., ['0180', '0181', ...])
        """
        try:
            # We need to fetch the variable definition for 'Kommun'
            url = f"{self.base_url}{config.ENDPOINT_CATEGORIES}"
            response = self.session.get(
                url,
                cert=self.cert_path,
                verify=config.VERIFY_SSL,
                timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            # Find 'Kommun' category
            municipalities = []
            for category in data:
                cat_id = category.get('Id_Kategori_AE', '')
                if cat_id == 'Kommun':
                    for value in category.get('VardeLista', []):
                        code = value.get('Varde', '')
                        # Municipality codes start with county code
                        if code.startswith(county_code) and len(code) == 4:
                            municipalities.append(code)
                    break
            
            if not municipalities:
                logger.warning(f"No municipalities found for county {county_code}")
                
            return sorted(municipalities)
            
        except Exception as e:
            logger.error(f"Failed to fetch municipalities: {e}")
            return []

    def get_legal_forms(self) -> List[str]:
        """Fetch all legal form codes (Juridisk form)."""
        try:
            url = f"{self.base_url}{config.ENDPOINT_CATEGORIES}"
            response = self.session.get(
                url, cert=self.cert_path, verify=config.VERIFY_SSL, timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            for category in data:
                if category.get('Id_Kategori_AE') == 'Juridisk form':
                    return [v.get('Varde') for v in category.get('VardeLista', [])]
            return []
        except Exception:
            return []

    def get_sni_codes(self) -> List[str]:
        """Fetch all 2-digit SNI codes (2-siffrig bransch 1)."""
        try:
            url = f"{self.base_url}{config.ENDPOINT_CATEGORIES}"
            response = self.session.get(
                url, cert=self.cert_path, verify=config.VERIFY_SSL, timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            for category in data:
                if category.get('Id_Kategori_AE') == '2-siffrig bransch 1':
                    return [v.get('Varde') for v in category.get('VardeLista', [])]
            return []
        except Exception:
            return []



# Data Transformer
# --------------------------------

class DataTransformer:
    """
    Transforms SCB API responses to LocationBasedForecasting schema.
    """

    def __init__(self):
        """Initialize data transformer."""
        self.employee_size_mapping = self._create_employee_mapping()

    def _create_employee_mapping(self) -> Dict[str, int]:
        """
        Create mapping from employee size class codes to midpoint values.

        SCB uses size class codes like '0', '1-4', '5-9', '10-19', etc.
        We convert these to midpoint integers.
        """
        return {
            '0': 0,
            '1': 1,
            '1-4': 2,
            '2-4': 3,
            '5-9': 7,
            '10-19': 15,
            '20-49': 35,
            '50-99': 75,
            '100-199': 150,
            '200-499': 350,
            '500-999': 750,
            '1000-': 1500,  # 1000+
            '1000-1499': 1250,
            '1500-': 2000,   # 1500+
            
            # API Codes for "Anställda" variable
            '0': 0,    # Uppgift saknas / 0 ? Check
            '1': 0,    # 0 anställda
            '2': 3,    # 1-4 anställda (midpoint 2.5 -> 3)
            '3': 7,    # 5-9 anställda
            '4': 15,   # 10-19 anställda
            '5': 35,   # 20-49 anställda
            '6': 75,   # 50-99 anställda
            '7': 150,  # 100-199 anställda
            '8': 350,  # 200-499 anställda
            '9': 750,  # 500-999 anställda
            '10': 1250, # 1000-1499
            '11': 1750, # 1500-1999
            '12': 2500, # 2000-2999
            '13': 3500, # 3000-3999
            '14': 4500, # 4000-4999
            '15': 7500, # 5000-9999
            '16': 15000 # 10000+
        }

    def _parse_employee_count(self, size_class: str) -> Optional[int]:
        """
        Parse employee count from size class string.

        Handles multiple formats:
        - Codes: '1', '2', '3', etc.
        - Ranges: '10-19', '5-9'
        - Swedish text: '1-4 anställda', '10-19 anställda'

        Args:
            size_class: Size class code or range (e.g., '10-19', '1-4 anställda')

        Returns:
            Midpoint employee count, or None if cannot parse
        """
        # CHECK FOR EMPTY INPUT
        if not size_class or size_class.strip() == '':
            return None

        # CLEAN INPUT
        size_class = size_class.strip()
        size_class = size_class.replace(' anställda', '').replace(' anstallda', '').strip()

        # TRY DIRECT MAPPING
        if size_class in self.employee_size_mapping:
            return self.employee_size_mapping[size_class]

        # TRY PARSING AS INTEGER
        try:
            return int(size_class)
        except ValueError:
            pass

        # TRY PARSING AS RANGE
        if '-' in size_class:
            parts = size_class.split('-')
            try:
                if len(parts) == 2:
                    low = int(parts[0])
                    high = int(parts[1]) if parts[1] else low * 2
                    return (low + high) // 2
            except ValueError:
                pass

        logger.warning(f"Could not parse employee count: {size_class}")
        return None

    def transform_supermarket(self, item: Dict) -> Optional[Dict]:
        """
        Transform SCB supermarket record to LocationBasedForecasting schema.

        Args:
            item: Raw SCB API response item

        Returns:
            Transformed dictionary, or None if invalid/incomplete
        """
        try:
            # EXTRACT COORDINATES
            north = item.get('Adress Nordkoord SWEREF99', '')
            east = item.get('Adress Ostkoord SWEREF99', '')

            # CHECK FOR MISSING COORDINATES
            if not north or not east or north == '*' or east == '*':
                logger.warning(f"Missing coordinates for {item.get('CFARNr')}")
                return None

            # CONVERT TO FLOAT
            try:
                north_float = float(north)
                east_float = float(east)
            except ValueError:
                logger.warning(f"Invalid coordinate values: north={north}, east={east}")
                return None

            # CONVERT TO WGS84
            lat, lon = sweref99_to_wgs84(north_float, east_float)

            if lat is None or lon is None:
                return None

            # EXTRACT OTHER FIELDS
            store_id = str(item.get('CFARNr', ''))
            store_name = item.get('Benämning', item.get('Företagsnamn', 'Unknown'))

            # RETURN TRANSFORMED RECORD
            return {
                'store_id': store_id,
                'store_name': store_name,
                'latitude': lat,
                'longitude': lon,
                'place_id': None,  # Not applicable for SCB
                'address': item.get('BesöksAdress', ''),
                'city': item.get('BesöksPostOrt', ''),
                'postal_code': item.get('BesöksPostNr', ''),
                'sni_code': item.get('Bransch_1, kod', '').strip(),
                'sni_description': item.get('Bransch_1', ''),
                'employee_count': self._parse_employee_count(
                    item.get('Storleksklass', '') or item.get('Stkl, kod', '')
                ),
                'source': 'SCB'
            }

        except Exception as e:
            logger.error(f"Error transforming supermarket record: {e}")
            return None

    def transform_company(self, item: Dict) -> Optional[Dict]:
        """
        Transform SCB company record to LocationBasedForecasting schema.

        Args:
            item: Raw SCB API response item

        Returns:
            Transformed dictionary, or None if invalid/incomplete
        """
        try:
            # EXTRACT COORDINATES
            north = item.get('Adress Nordkoord SWEREF99', '')
            east = item.get('Adress Ostkoord SWEREF99', '')

            # CHECK FOR MISSING COORDINATES
            if not north or not east or north == '*' or east == '*':
                return None

            # CONVERT TO FLOAT
            try:
                north_float = float(north)
                east_float = float(east)
            except ValueError:
                return None

            # CONVERT TO WGS84
            lat, lon = sweref99_to_wgs84(north_float, east_float)

            if lat is None or lon is None:
                return None

            # PARSE EMPLOYEE COUNT
            employee_count = self._parse_employee_count(
                item.get('Storleksklass', '') or item.get('Stkl, kod', '') or item.get('Anställda', '')
            )

            # FILTER BY EMPLOYEE THRESHOLD
            if employee_count is None or employee_count < config.MIN_EMPLOYEE_COUNT:
                return None

            # EXTRACT OTHER FIELDS
            company_id = str(item.get('CFARNr', ''))
            company_name = item.get('Benämning', item.get('Företagsnamn', 'Unknown'))
            sni_code = item.get('Bransch_1, kod', '').strip()

            # EXCLUDE FOOD RETAIL
            if sni_code and any(sni_code.startswith(prefix) for prefix in config.EXCLUDE_SNI_PREFIX):
                return None

            # RETURN TRANSFORMED RECORD
            return {
                'company_id': company_id,
                'company_name': company_name,
                'latitude': lat,
                'longitude': lon,
                'employee_count': employee_count,
                'address': item.get('BesöksAdress', ''),
                'city': item.get('BesöksPostOrt', ''),
                'postal_code': item.get('BesöksPostNr', ''),
                'sni_code': sni_code,
                'sni_description': item.get('Bransch_1', ''),
                'county_code': item.get('Län, kod', ''),
                'municipality_code': item.get('Kommun, kod', ''),
                'source': 'SCB'
            }

        except Exception as e:
            logger.error(f"Error transforming company record: {e}")
            return None


# Fetchers
# --------------------------------

class SupermarketFetcher:
    """
    Fetches supermarket data from SCB API.
    """

    def __init__(self, api_client: SCBAPIClient, transformer: DataTransformer):
        """
        Initialize supermarket fetcher.

        Args:
            api_client: SCB API client
            transformer: Data transformer
        """
        self.api_client = api_client
        self.transformer = transformer

    def fetch_all(self) -> List[Dict]:
        """
        Fetch all supermarket/food retail data for the entire country.
        
        Iterates through counties to avoid hitting API limits.
        """
        all_supermarkets = []
        
        if config.TEST_MODE:
            logger.info(f"TEST MODE ENABLED: Checking Stockholm County only")
            target_counties = ['01'] 
        else:
            target_counties = config.COUNTIES

        for i, county_code in enumerate(target_counties, 1):
            county_name = config.COUNTY_NAMES.get(county_code, county_code)
            logger.info(f"  [{i}/{len(target_counties)}] Processing {county_name} (län {county_code})...")
            
            county_stores = self._fetch_county(county_code)
            all_supermarkets.extend(county_stores)
            logger.info(f"      → {len(county_stores)} stores")
            
            if config.TEST_MODE:
                break

        if config.TEST_MODE:
            logger.info("TEST MODE: Skipping Supermarket fetch to focus on Company logic")
            return []

        logger.info(f"  ✓ Total supermarkets fetched: {len(all_supermarkets)}")
        return all_supermarkets

    def _fetch_county(self, county_code: str) -> List[Dict]:
        """
        Fetch supermarkets for a single county.
        Splits by municipality if count > 2000.
        """
        # Base payload
        payload = {
            "Arbetsställestatus": "1",
            "Kategorier": [
                {
                    "Kategori": "Bransch", 
                    "Kod": config.FOOD_RETAIL_CODES,
                    "BranschNiva": 3
                },
                {"Kategori": "Län", "Kod": [county_code]}
            ]
        }
        
        count = self.api_client.count_establishments(payload)
        logger.info(f"      Total stores in county {county_code}: {count}")
        
        if count <= config.MAX_RESULTS_PER_REQUEST:
            raw_data = self.api_client.fetch_establishments(payload)
            return self._transform_and_filter(raw_data)
            
        # If > 2000, split by municipality
        logger.info(f"      Count {count} > {config.MAX_RESULTS_PER_REQUEST}, splitting by Municipality...")
        
        all_stores = []
        municipalities = self.api_client.get_municipalities(county_code)
        
        for muni_code in municipalities:
            muni_payload = {
                "Arbetsställestatus": "1",
                "Kategorier": [
                    {
                        "Kategori": "Bransch", 
                        "Kod": config.FOOD_RETAIL_CODES,
                        "BranschNiva": 3
                    },
                    {"Kategori": "Kommun", "Kod": [muni_code]}
                ]
            }
            
            muni_count = self.api_client.count_establishments(muni_payload)
            if muni_count > 0:
                if muni_count > config.MAX_RESULTS_PER_REQUEST:
                     # Very unlikely for supermarkets, but safety check
                     logger.warning(f"Municipality {muni_code} has {muni_count} stores! Truncation will occur.")
                
                raw_data = self.api_client.fetch_establishments(muni_payload)
                all_stores.extend(self._transform_and_filter(raw_data))
                
        return all_stores

    def _transform_and_filter(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform raw data and filter out invalid/unwanted records.

        Args:
            raw_data: List of raw SCB records

        Returns:
            List of transformed and filtered dictionaries
        """
        transformed = []
        for item in raw_data:
            result = self.transformer.transform_supermarket(item)
            if result:
                transformed.append(result)
        logger.info(f"  ✓ Transformed {len(transformed)} supermarkets")
        return transformed


class CompanyFetcher:
    """
    Fetches company data from SCB API with pagination by county.
    """

    def __init__(self, api_client: SCBAPIClient, transformer: DataTransformer):
        """
        Initialize company fetcher.

        Args:
            api_client: SCB API client
            transformer: Data transformer
        """
        self.api_client = api_client
        self.transformer = transformer

    def fetch_all(self) -> List[Dict]:
        """
        Fetch all companies (≥ MIN_EMPLOYEE_COUNT employees, excluding food retail).

        Uses pagination by county to handle large result sets.

        Returns:
            List of transformed company dictionaries
        """
        logger.info(f"Fetching companies (≥{config.MIN_EMPLOYEE_COUNT} employees)...")

        all_companies = []

        if config.TEST_MODE:
            logger.info(f"TEST MODE ENABLED: Checking Stockholm County only")
            target_counties = ['01'] # Force Stockholm for test 
        else:
            target_counties = config.COUNTIES

        for i, county_code in enumerate(target_counties, 1):
            county_name = config.COUNTY_NAMES.get(county_code, county_code)
            logger.info(f"  [{i}/{len(target_counties)}] Processing {county_name} (län {county_code})...")

            companies = self._fetch_county(county_code)
            all_companies.extend(companies)
            logger.info(f"      → {len(companies)} companies")
            
            if config.TEST_MODE:
                break

        logger.info(f"  ✓ Total companies fetched: {len(all_companies)}")
        return all_companies

    def _fetch_county(self, county_code: str) -> List[Dict]:
        """
        Fetch companies for a single county.
        
        Uses cascading pagination strategies:
        1. Fetch whole county
        2. If > limit, fetch by Municipality
        3. If Municipality > limit, fetch by Employee Size Class
        
        Args:
            county_code: Two-digit county code
            
        Returns:
            List of transformed company dictionaries
        """
        # STRATEGY 1: Fetch whole county
        payload = {
            "Arbetsställestatus": "1",
            "Kategorier": [{"Kategori": "Län", "Kod": [county_code]}]
        }
        
        count = self.api_client.count_establishments(payload)
        logger.info(f"      Total establishments in county {county_code}: {count}")

        if count <= config.MAX_RESULTS_PER_REQUEST:
            raw_data = self.api_client.fetch_establishments(payload)
            return self._transform_and_filter(raw_data)
            
        # STRATEGY 2: Split by Municipality
        logger.info(f"      Count {count} > {config.MAX_RESULTS_PER_REQUEST}, splitting by Municipality...")
        
        all_companies = []
        municipalities = self.api_client.get_municipalities(county_code)
        
        if not municipalities:
            logger.error(f"No municipalities found for county {county_code}! Cannot paginate.")
            return []
            
        logger.info(f"Found {len(municipalities)} municipalities to process")
        
        for muni_code in municipalities:
            muni_companies = self._fetch_municipality(muni_code)
            all_companies.extend(muni_companies)
            
        return all_companies

    def _fetch_municipality(self, muni_code: str) -> List[Dict]:
        """
        Fetch companies for a single municipality.
        fallback to size class splitting if needed.
        """
        payload = {
            "Arbetsställestatus": "1",
            "Kategorier": [{"Kategori": "Kommun", "Kod": [muni_code]}]
        }
        
        count = self.api_client.count_establishments(payload)
        # logger.debug(f"        Municipality {muni_code}: {count} establishments")
        
        if count <= config.MAX_RESULTS_PER_REQUEST:
            if count > 0:
                raw_data = self.api_client.fetch_establishments(payload)
                return self._transform_and_filter(raw_data)
            return []
            
        # STRATEGY 3: Split by Size Class (within Municipality)
        logger.info(f"Municipality {muni_code} has {count} records (> {config.MAX_RESULTS_PER_REQUEST}). Splitting by Size Class...")
        return self._fetch_by_size_class(location_type="Kommun", location_code=muni_code)

    def _fetch_by_size_class(self, location_type: str, location_code: str) -> List[Dict]:
        """
        Fetch companies by iterating through employee size classes.
        
        Args:
            location_type: "Län" or "Kommun"
            location_code: The code for the location
        """
        all_companies = []
        # Codes 4-16 correspond to companies with >= 10 employees
        target_size_classes = [str(i) for i in range(4, 17)]
        
        for size_code in target_size_classes:
            payload = {
                "Arbetsställestatus": "1",
                "Kategorier": [
                    {"Kategori": location_type, "Kod": [location_code]},
                    {"Kategori": "Anställda", "Kod": [size_code]}
                ]
            }
            
            count = self.api_client.count_establishments(payload)
            
            if count > config.MAX_RESULTS_PER_REQUEST:
                logger.info(f"Size class {size_code} in {location_type} {location_code} has {count} records! Splitting by SNI (Level 4)...")
                sni_companies = self._fetch_by_sni(location_type, location_code, size_code)
                all_companies.extend(sni_companies)
                continue
                
            if count > 0:
                raw_slice = self.api_client.fetch_establishments(payload)
                transformed_slice = self._transform_and_filter(raw_slice)
                all_companies.extend(transformed_slice)
                
        return all_companies

    def _fetch_by_sni(self, location_type: str, location_code: str, size_code: str) -> List[Dict]:
        """
        Level 4 Pagination: Split by SNI Industry (2-digit).
        Replaces Legal Form split due to API license restrictions.
        """
        all_companies = []
        sni_codes = self.api_client.get_sni_codes()
        
        # Skip Retail (47) and Restaurants (56) to save requests
        relevant_sni = [c for c in sni_codes if not (c.startswith('47') or c.startswith('56'))]
        
        for sni_code in relevant_sni:
            payload = {
                "Arbetsställestatus": "1",
                "Kategorier": [
                    {"Kategori": location_type, "Kod": [location_code]},
                    {"Kategori": "Anställda", "Kod": [size_code]},
                    {"Kategori": "2-siffrig bransch 1", "Kod": [sni_code]}
                ]
            }
            
            count = self.api_client.count_establishments(payload)
            
            if count > config.MAX_RESULTS_PER_REQUEST:
                 logger.warning(f"SNI {sni_code} (Size {size_code}) has {count} records! Truncation inevitable.")
            
            if count > 0:
                raw_data = self.api_client.fetch_establishments(payload)
                all_companies.extend(self._transform_and_filter(raw_data))
                
        return all_companies

    def _transform_and_filter(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Transform raw data and filter out invalid/unwanted records.

        Args:
            raw_data: List of raw SCB records

        Returns:
            List of transformed and filtered dictionaries
        """
        transformed = []
        for item in raw_data:
            result = self.transformer.transform_company(item)
            if result:
                transformed.append(result)
        return transformed


# Parquet Writer
# --------------------------------
class ParquetWriter:
    """
    Writes processed data to parquet files.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize parquet writer.

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def write_supermarkets(self, records: List[Dict]) -> Path:
        """
        Write supermarket data to parquet file.

        Args:
            records: List of supermarket dictionaries

        Returns:
            Path to output file
        """
        # CHECK FOR EMPTY RECORDS
        if not records:
            logger.warning("No supermarket records to write!")
            return None

        # CONVERT TO DATAFRAME
        df = pd.DataFrame(records)

        # VALIDATE REQUIRED COLUMNS
        required_cols = ['store_id', 'store_name', 'latitude', 'longitude']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # SET DATA TYPES
        df['store_id'] = df['store_id'].astype(str)
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)

        # REMOVE DUPLICATES
        original_count = len(df)
        df = df.drop_duplicates(subset=['store_id'])
        if len(df) < original_count:
            logger.warning(f"Removed {original_count - len(df)} duplicate supermarkets")

        # WRITE TO PARQUET
        output_path = self.output_dir / config.SUPERMARKETS_FILE
        df.to_parquet(output_path, index=False)
        logger.info(f"✓ Wrote {len(df)} supermarkets to {output_path}")

        return output_path

    def write_companies(self, records: List[Dict]) -> Path:
        """
        Write company data to parquet file.

        Args:
            records: List of company dictionaries

        Returns:
            Path to output file
        """
        # CHECK FOR EMPTY RECORDS
        if not records:
            logger.warning("No company records to write!")
            return None

        # CONVERT TO DATAFRAME
        df = pd.DataFrame(records)

        # VALIDATE REQUIRED COLUMNS
        required_cols = ['company_id', 'company_name', 'latitude', 'longitude', 'employee_count']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # SET DATA TYPES
        df['company_id'] = df['company_id'].astype(str)
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df['employee_count'] = df['employee_count'].astype(int)

        # REMOVE DUPLICATES
        original_count = len(df)
        df = df.drop_duplicates(subset=['company_id'])
        if len(df) < original_count:
            logger.warning(f"Removed {original_count - len(df)} duplicate companies")

        # WRITE TO PARQUET
        output_path = self.output_dir / config.COMPANIES_FILE
        df.to_parquet(output_path, index=False)
        logger.info(f"✓ Wrote {len(df)} companies to {output_path}")

        return output_path



# Validation
# --------------------------------

def validate_output(supermarket_path: Optional[Path], company_path: Optional[Path]):
    """
    Validate output parquet files.

    Args:
        supermarket_path: Path to supermarket parquet file (or None if not written)
        company_path: Path to company parquet file (or None if not written)
    """
    logger.info("Validating output files...")

    if not supermarket_path or not supermarket_path.exists():
        logger.error("Supermarket file not found or not written!")
        return

    if not company_path or not company_path.exists():
        logger.error("Company file not found or not written!")
        return

    # Load files
    supermarkets = pd.read_parquet(supermarket_path)
    companies = pd.read_parquet(company_path)

    logger.info(f"  Supermarkets: {len(supermarkets)} records")
    logger.info(f"  Companies: {len(companies)} records")

    # Validate supermarkets
    assert all(col in supermarkets.columns for col in ['latitude', 'longitude', 'store_name', 'store_id']), \
        "Missing required supermarket columns"
    assert supermarkets['latitude'].between(config.SWEDEN_LAT_MIN, config.SWEDEN_LAT_MAX).all(), \
        "Invalid supermarket latitudes"
    assert supermarkets['longitude'].between(config.SWEDEN_LON_MIN, config.SWEDEN_LON_MAX).all(), \
        "Invalid supermarket longitudes"
    assert supermarkets['store_id'].nunique() == len(supermarkets), \
        "Duplicate supermarket IDs found"

    # Validate companies
    assert all(col in companies.columns for col in ['latitude', 'longitude', 'employee_count', 'company_id']), \
        "Missing required company columns"
    assert companies['latitude'].between(config.SWEDEN_LAT_MIN, config.SWEDEN_LAT_MAX).all(), \
        "Invalid company latitudes"
    assert companies['longitude'].between(config.SWEDEN_LON_MIN, config.SWEDEN_LON_MAX).all(), \
        "Invalid company longitudes"
    assert companies['employee_count'].min() >= config.MIN_EMPLOYEE_COUNT, \
        f"Companies below minimum employee threshold found"
    assert companies['company_id'].nunique() == len(companies), \
        "Duplicate company IDs found"

    logger.info("  ✓ All validations passed!")

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Supermarkets by SNI code:")
    for sni_code, count in supermarkets['sni_code'].value_counts().head(5).items():
        desc = config.SNI_CODE_DESCRIPTIONS.get(sni_code, 'Unknown')
        logger.info(f"    {sni_code} ({desc}): {count}")

    logger.info(f"\n  Companies by county:")
    for county_code, count in companies['county_code'].value_counts().head(5).items():
        county_name = config.COUNTY_NAMES.get(county_code, county_code)
        logger.info(f"    {county_code} ({county_name}): {count}")

    logger.info(f"\n  Company employee distribution:")
    logger.info(f"    Mean: {companies['employee_count'].mean():.0f}")
    logger.info(f"    Median: {companies['employee_count'].median():.0f}")
    logger.info(f"    Min: {companies['employee_count'].min()}")
    logger.info(f"    Max: {companies['employee_count'].max()}")



# Main
# --------------------------------

def main():
    """Main execution function."""
    # PRINT HEADER
    print("SCB Data Fetcher for LocationBasedForecasting")

    start_time = time.time()

    try:
        # STEP 1: INITIALIZE COMPONENTS
        logger.info("\n[Step 1/5] Initializing components...")
        api_client = SCBAPIClient(cert_path=config.CERT_PATH)
        transformer = DataTransformer()
        writer = ParquetWriter(output_dir=config.get_absolute_output_dir())

        # STEP 2: FETCH SUPERMARKET DATA
        logger.info("\n[Step 2/5] Fetching supermarket data...")
        supermarket_fetcher = SupermarketFetcher(api_client, transformer)
        supermarkets = supermarket_fetcher.fetch_all()

        if not supermarkets:
            logger.error("No supermarkets fetched! Aborting.")
            return 1

        # STEP 3: FETCH COMPANY DATA
        logger.info(f"\n[Step 3/5] Fetching company data (≥{config.MIN_EMPLOYEE_COUNT} employees)...")
        company_fetcher = CompanyFetcher(api_client, transformer)
        companies = company_fetcher.fetch_all()

        if not companies:
            logger.error("No companies fetched! Aborting.")
            return 1

        # STEP 4: WRITE TO PARQUET FILES
        logger.info("\n[Step 4/5] Writing parquet files...")
        supermarket_path = writer.write_supermarkets(supermarkets)
        company_path = writer.write_companies(companies)

        # STEP 5: VALIDATE OUTPUT
        logger.info("\n[Step 5/5] Validating output...")
        validate_output(supermarket_path, company_path)

        # PRINT SUCCESS SUMMARY
        elapsed = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("SUCCESS")
        logger.info("="*80)
        logger.info(f"Supermarkets: {len(supermarkets)} records → {supermarket_path}")
        logger.info(f"Companies: {len(companies)} records → {company_path}")
        logger.info(f"Elapsed time: {elapsed/60:.1f} minutes")
        logger.info(f"Log file: {log_file}")
        logger.info("="*80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n\nFATAL ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
