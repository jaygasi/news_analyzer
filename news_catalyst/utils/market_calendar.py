from datetime import datetime, timedelta
import pandas as pd
import pandas_market_calendars as mcal


class MarketCalendar:
    """
    MarketCalendar provides various functionalities to interact with market calendars.

    Attributes:
        exchange (Exchange): The exchange for which the calendar is created.
    """

    def __init__(self, exchange: str):
        """
        Initializes the MarketCalendar with a specific exchange.

        Parameters:
            exchange (Exchange): The exchange for which the calendar is created.
        """
        self.exchange = exchange
        self.calendar = mcal.get_calendar(exchange)

    def get_timezone(self) -> str:
        """
        Gets the time zone - COMPATIBILITY FIXED
        """
        try:
            if hasattr(self.calendar.tz, 'zone'):
                return self.calendar.tz.zone
            elif hasattr(self.calendar.tz, 'key'):
                return self.calendar.tz.key  
            else:
                return str(self.calendar.tz)
        except Exception:
            return "US/Eastern"  # Safe fallback

    def get_last_holidays(self, num_holidays: int = 5) -> pd.DatetimeIndex:
        """
        Gets the last n holidays of the exchange calendar.

        Parameters:
            num_holidays (int): The number of holidays to retrieve.

        Returns:
            pd.DatetimeIndex: The list of last holidays as pandas Timestamps.
        """
        holidays = self.calendar.holidays()
        last_holidays = holidays.holidays[-num_holidays:]
        return pd.to_datetime(last_holidays)

    def get_market_times(self) -> dict:
        """
        Gets the regular market times of the exchange calendar.

        Returns:
            dict: A dictionary containing market times.
        """
        return self.calendar.regular_market_times

    def get_open_business_days(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Gets the open valid business days for the exchange between the given dates.

        Parameters:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pd.DatetimeIndex: The list of open business days.
        """
        return self.calendar.valid_days(start_date=start_date, end_date=end_date)

    def get_market_schedule(self, start_date: str, end_date: str, start: str = "regular", end: str = "regular") -> pd.DataFrame:
        """
        Gets the market schedule for the exchange between the given dates.

        Parameters:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            start (str): The start time (e.g., "pre", "regular", "post").
            end (str): The end time (e.g., "pre", "regular", "post").

        Returns:
            pd.DataFrame: The market schedule.
        """
        return self.calendar.schedule(start_date=start_date, end_date=end_date, start=start, end=end)

    def get_early_closes(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Gets the early closes for the exchange between the given dates.

        Parameters:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: The early close schedule.
        """
        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
        return self.calendar.early_closes(schedule=schedule)

    def is_market_open_now(self, extended_hours: bool = False) -> bool:
        """
        Checks if the market is open right now, optionally including extended hours.

        Parameters:
            extended_hours (bool): Whether to consider pre- and post-market hours.

        Returns:
            bool: True if the market is open, False otherwise.
        """
        now_timestamp = pd.Timestamp(datetime.now(), tz=self.get_timezone())
        start_timestamp = pd.Timestamp(datetime.now() - timedelta(days=1), tz=self.get_timezone())
        end_timestamp = pd.Timestamp(datetime.now() + timedelta(days=1), tz=self.get_timezone())

        try:
            # Fetch the schedule
            if extended_hours:
                schedule = self.calendar.schedule(
                    start_date=start_timestamp.strftime('%Y-%m-%d'),
                    end_date=end_timestamp.strftime('%Y-%m-%d'),
                    start="pre",
                    end="post"
                )
            else:
                schedule = self.calendar.schedule(
                    start_date=start_timestamp.strftime('%Y-%m-%d'),
                    end_date=end_timestamp.strftime('%Y-%m-%d')
                )

            # Ensure now_timestamp is within the schedule range
            if schedule.empty:
                raise ValueError("No schedule available for the specified range.")

            # Iterate through the schedule to check if now is within any open period
            for idx, row in schedule.iterrows():
                if row["market_open"] <= now_timestamp <= row["market_close"]:
                    return True
                if extended_hours and row["pre"] <= now_timestamp <= row["post"]:
                    return True

            # If no match, market is not open
            return False

        except ValueError as e:
            print(f"Error: {e}")
            print(f"Now Timestamp: {now_timestamp}")
            print(f"Schedule Start: {start_timestamp.strftime('%Y-%m-%d')}")
            print(f"Schedule End: {end_timestamp.strftime('%Y-%m-%d')}")
            print(f"Schedule: {schedule}")
            return False