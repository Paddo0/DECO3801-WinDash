import constants
import datetime
from helpers.DatabaseMutationHelper import *
from helpers.DatabaseRetrieveHelper import *
from helpers.DatabaseUpdateHelper import *
from helpers.CsvDataHelper import *


def SetupWithHistoricalData(db, meterId, start_time):
    """
    Sets up the virtual meter with historical data. This function clears any existing data, adds a new meter,
    and populates the meter with historical daily, yesterday's, and overall summary data. It also sets the 
    current day and prepares the meter for future data insertion.
    :param db: The reference to the database.
    :param meterId: The ID of the meter being set up.
    :param start_time: A `datetime` object representing the starting time from which the meter is configured.
    :return: None
    """

    # Clear any existing data in the database for the given meter
    ClearAllData(db, meterId)

    # Add the meter to the database with initial setup
    AddMeter(db, meterId)

    # Extract historical data from the CSV for processing (yesterday's data, current day's data, overall summary, and future data)
    yesterday_daily_data, daily_data, overall_data = ExtractPastVirtualMeterCsvData(constants.dataFilepath, meterId)

    # Set the current day in the meter's daily data to the start time's date
    SetCurrentDay(db, meterId, start_time.date())

    # Add the minute-level data for the current day up to the start time
    AddMinuteArray(db, meterId, daily_data)

    # Set yesterday's date and add yesterday's minute-level data to the previousDayDailyData collection
    yesterday_date = (start_time - timedelta(days=1)).date()  # Calculate yesterday's date
    SetYesterdayDay(db, meterId, yesterday_date)
    AddYesterdayMinuteArray(db, meterId, yesterday_daily_data)

    # Add the overall summary of historical daily data to the overallData collection
    AddDayArraySummary(db, meterId, overall_data)

def setupWithNoData(db, meterId, start_time):
    """
    Sets up a virtual meter without any historical data. This function clears any existing data for the meter,
    initializes a new meter with empty data, and sets the current day and the previous day's date.
    :param db: The reference to the database where the meter data is stored.
    :param meterId: The ID of the meter to be set up.
    :param start_time: A `datetime` object representing the starting time from which the meter setup will begin.
    :return: None
    """
    
    # Clear any existing data for the specified meter in the database
    ClearAllData(db, meterId)

    # Add the new meter with default setup to the database
    AddMeter(db, meterId)

    # Set the current day in the dailyData collection to the start time's date
    SetCurrentDay(db, meterId, start_time.date())

    # Calculate yesterday's date by subtracting one day from the start time
    yesterday_date = (start_time - timedelta(days=1)).date()

    # Set the calculated yesterday date in the previousDayDailyData collection
    SetYesterdayDay(db, meterId, yesterday_date)

def StartFromPoint(db, meterId, start_time):
    """
    Sets up the virtual meter starting from a given point in time. It retrieves any missing data between the latest 
    time recorded and up to and including the specified start time, processes that data, and then resumes the meter from the next minute 
    after the start time. The function performs the following:
    - Retrieves the latest recorded minute data from the database.
    - Extracts missing data between the latest time and the specified start time.
    - Adds minute-level data to the database.
    - If the missing data spans multiple days, it saves daily summaries and manages the transition to the new day.
    - Begins running the meter after the specified start time, processing future data.
    :param db: The reference to the database where the meter data is stored.
    :param meterId: The ID of the meter.
    :param start_time: A `datetime` object representing the time from which to start the virtual meter.
    :return: None
    """

    # Retrieve the latest minute data from the database
    latest_time, _, _ = GetLatestMinute(db, meterId)

    # Remove timezone awareness from latest_time and start_time if present
    if latest_time.tzinfo is not None:
        latest_time = latest_time.replace(tzinfo=None)
    
    if start_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=None)

    # If latest_time is greater than or equal to start_time, print an error and return
    if latest_time >= start_time:
        print(f"Error: start_time {start_time} must be after the latest_time {latest_time}.")
        return

    # Extract the range of data between the latest recorded time and the start time
    missing_data_range = ExtractRangeVirtualMeterCsvData(constants.dataFilepath, latest_time, start_time)

    # If the missing data is from the same day
    if latest_time.date() == start_time.date():
        AddMinuteArray(db, meterId, missing_data_range)

    # If the start date is at most one day after the latest recorded time
    elif abs((latest_time.date() - start_time.date()).days) <= 1:
        remaining_daily_data = []
        new_daily_data = []

        # Loop through the missing data to separate data for the current day and the next day
        for entry in missing_data_range:
            entry_time = entry[0]

            # Ensure the entry time is timezone-naive
            if entry_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=None)

            # Append data to remaining_daily_data if it belongs to the latest time's date
            if entry_time.date() == latest_time.date():
                remaining_daily_data.append(entry)
            # Otherwise, append it to new_daily_data for the next day
            else:
                new_daily_data.append(entry)
        
        # Add the remaining data for the current day
        AddMinuteArray(db, meterId, remaining_daily_data)
        
        # Calculate and save the overall summary for the current day and move data to yesterday's series
        CalculateAndSaveOverallData(db, meterId)

        # Add the new day's minute-level data
        AddMinuteArray(db, meterId, new_daily_data)

    # If there is more than a day's difference between latest_time and start_time
    else:
        # Get the daily data from the current day
        latest_daily_data = GetAllDaily(db, meterId)  # Retrieve the latest daily data

        # Dictionary to store daily data for all days between the latest time and the day before start time
        historical_daily_data_dict = {}
        historical_daily_data_dict[latest_time.date()] = latest_daily_data

        new_daily_data = []
        new_yesterday_daily_data = []
        overall_data = []

        # Loop through the missing data range and distribute data into respective days
        for entry in missing_data_range:
            entry_time = entry[0]

            # Ensure the entry time is timezone-naive
            if entry_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=None)

            # Append data to the current day
            if entry_time.date() == latest_time.date():
                historical_daily_data_dict[latest_time.date()].append(entry)
            # Append data to the new days
            else:
                if entry_time.date() not in historical_daily_data_dict:
                    historical_daily_data_dict[entry_time.date()] = []
                historical_daily_data_dict[entry_time.date()].append(entry)

                # If the entry is for the day before the start date
                if abs((latest_time.date() - entry_time.date()).days) <= 1:
                    new_yesterday_daily_data.append(entry)

        # Calculate the daily summary for all dates in the historical data
        for date, date_data in historical_daily_data_dict.items():
            day_summary = CalculateDaySummary(date_data)
            overall_data.append((date, day_summary))

    # Extract the future data after the start time
    future_data = ExtractFutureVirtualMeterCsvData(constants.dataFilepath, start_time)

    # Start running the virtual meter from the start time
    Run(db, meterId, future_data)


import datetime

def Resume(db, meterId):
    """
    Resumes the virtual meter from the most recent recorded time. If no previous data is found, it starts from the 
    current time. The function retrieves future data starting from the latest time and processes it.
    :param db: The reference to the database where the meter data is stored.
    :param meterId: The ID of the meter.
    :return: None
    """

    # Get the latest recorded minute data from the database
    latest_time, _, _ = GetLatestMinute(db, meterId)
    
    # If there is no latest data, set the latest time to the current time (rounded down to the nearest minute)
    if latest_time is None:
        latest_time = (datetime.now()).replace(second=0, microsecond=0)

    # Extract all future data starting from after the latest time
    future_data = ExtractFutureVirtualMeterCsvData(constants.dataFilepath, latest_time)
    
    # Start running the virtual meter from the latest time with the extracted future data
    Run(db, meterId, future_data)


def Run(db , meterId, future_data):
    print(future_data)
    return