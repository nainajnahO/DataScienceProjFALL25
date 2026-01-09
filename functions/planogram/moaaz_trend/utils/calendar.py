"""
Swedish Holiday Calendar
========================

Utility class for managing Swedish holidays and working days.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
import logging

from ..config import HOLIDAYS_FILE

logger = logging.getLogger(__name__)


class SwedishHolidayCalendar:
    """
    Manages Swedish holidays and working days calculations.
    """
    
    def __init__(self, holidays_file: Optional[Path] = None):
        """
        Initialize calendar with Swedish holidays.
        
        Args:
            holidays_file: Path to CSV with Swedish holidays. If None, uses HOLIDAYS_FILE from config.
        """
        if holidays_file is None:
            holidays_file = HOLIDAYS_FILE
        
        self.holidays_file = holidays_file
        self.holidays_df = None
        self.holiday_set = set()
        
        self._load_holidays()
    
    def _load_holidays(self) -> None:
        """Load holidays from CSV file."""
        if not self.holidays_file.exists():
            logger.warning(f"Holidays file not found: {self.holidays_file}")
            return
        
        try:
            self.holidays_df = pd.read_csv(self.holidays_file)
            
            # Parse dates - format is "DD-MMM"
            if 'Date' in self.holidays_df.columns and 'Year' in self.holidays_df.columns:
                # Combine Year and Date to create full date
                self.holidays_df['FullDate'] = self.holidays_df['Year'].astype(str) + '-' + self.holidays_df['Date']
                self.holidays_df['FullDate'] = pd.to_datetime(self.holidays_df['FullDate'], format='%Y-%d-%b')
                self.holiday_set = set(self.holidays_df['FullDate'].dt.date)
            
            logger.info(f"Loaded {len(self.holiday_set)} Swedish holidays")
        except Exception as e:
            logger.error(f"Failed to load holidays from {self.holidays_file}: {e}")
    
    def is_holiday(self, check_date: date) -> bool:
        """
        Check if a given date is a Swedish holiday.
        
        Args:
            check_date: Date to check
            
        Returns:
            True if the date is a holiday
        """
        return check_date in self.holiday_set
    
    def get_holiday_name(self, check_date: date) -> Optional[str]:
        """
        Get the name of the holiday on a given date.
        
        Args:
            check_date: Date to check
            
        Returns:
            Holiday name if found, None otherwise
        """
        if self.holidays_df is None or 'Name' not in self.holidays_df.columns:
            return None
        
        # Use FullDate if available, otherwise try parsing original Date column
        if 'FullDate' in self.holidays_df.columns:
            matching = self.holidays_df[self.holidays_df['FullDate'].dt.date == check_date]
        else:
            check_dt = pd.to_datetime(check_date)
            matching = self.holidays_df[pd.to_datetime(self.holidays_df['Date']).dt.date == check_date]
        
        if len(matching) > 0:
            return matching['Name'].iloc[0]
        
        return None
    
    def count_working_days(self, start_date: date, end_date: date) -> int:
        """
        Count working days (non-holiday weekdays) between two dates.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Number of working days
        """
        working_days = 0
        current = start_date
        
        while current <= end_date:
            # Check if weekday (Monday=0, Sunday=6)
            if current.weekday() < 5:  # Monday-Friday
                if not self.is_holiday(current):
                    working_days += 1
            
            current += timedelta(days=1)
        
        return working_days
    
    def get_holidays_in_range(self, start_date: date, end_date: date) -> List[Dict[str, str]]:
        """
        Get all holidays in a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of dictionaries with 'date' and 'name' keys
        """
        holidays = []
        current = start_date
        
        while current <= end_date:
            if self.is_holiday(current):
                holiday_name = self.get_holiday_name(current)
                holidays.append({
                    'date': current.strftime('%Y-%m-%d'),
                    'name': holiday_name or 'Swedish Holiday'
                })
            
            current += timedelta(days=1)
        
        return holidays

