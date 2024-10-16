import constants
import datetime
import time
from helpers.DatabaseMutationHelper import *
from helpers.DatabaseRetrieveHelper import *
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

    # Add the meter to the database with initial setup
    AddMeter(db, meterId)

    # Extract historical data from the CSV for processing (yesterday's data, current day's data, overall summary, and future data)
    yesterday_daily_data, daily_data, overall_data = ExtractPastVirtualMeterCsvData(constants.dataFilepath, start_time)

    # Set the current date in the meter's daily data to the start time's date (at midday since firebase datefield needs a time too)
    midday_start_time = start_time.replace(hour=12, minute=0, second=0, microsecond=0)
    SetCurrentDay(db, meterId, midday_start_time)

    # Add the minute-level data for the current day up to the start time
    AddMinuteArray(db, meterId, daily_data)

    # Set yesterday's date and add yesterday's minute-level data to the previousDayDailyData collection
    yesterday_midday_time = (start_time - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    SetYesterdayDay(db, meterId, yesterday_midday_time)
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

    # Add the new meter with default setup to the database
    AddMeter(db, meterId)

    # Set the current date in the meter's daily data to the start time's date (at midday since firebase datefield needs a time too)
    midday_start_time = start_time.replace(hour=12, minute=0, second=0, microsecond=0)
    SetCurrentDay(db, meterId, midday_start_time)

    # Set yesterday's date
    yesterday_date = start_time - timedelta(days=1)
    yesterday_midday_time = yesterday_date.replace(hour=12, minute=0, second=0, microsecond=0)
    SetYesterdayDay(db, meterId, yesterday_midday_time)

def StartFromPoint(db, meterId, start_time, run_fast):
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
    :param run_fast: A boolean flag that determines whether to use the fast simulation mode. If `True`, the `RunFast` function is used to add entries at a quicker pace based on the specified interval. If `False`, the standard `Run` function is used, simulating entries in real-time.
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
                if (entry_time.date() not in historical_daily_data_dict) and (entry_time.date() != start_time.date()):
                    historical_daily_data_dict[entry_time.date()] = []

                if (entry_time.date() != start_time.date()):
                    historical_daily_data_dict[entry_time.date()].append(entry)

                # If the entry is for the day before the start date
                #if abs((start_time.date() - entry_time.date()).days) <= 1:
                if (entry_time.date() == ((start_time - datetime.timedelta(days=1)).date())):
                    new_yesterday_daily_data.append(entry)

                if (entry_time.date() == start_time.date()):
                    new_daily_data.append(entry)

        # Calculate the daily summary for all dates in the historical data
        for date, date_data in historical_daily_data_dict.items():
            day_summary = CalculateDaySummary(date_data)
            midday_date = date_data[0][0].replace(hour=12, minute=0, second=0)
            overall_data.append((midday_date, day_summary))

        # Clear existing data from dailyData and previousDailyData series
        ClearTodayData(db, meterId)
        ClearYesterdayData(db, meterId)

        # Set the current day and yesterday's date
        midday_start_time = start_time.replace(hour=12, minute=0, second=0, microsecond=0)
        midday_yesterday_time = (start_time - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
        SetCurrentDay(db, meterId, midday_start_time)
        SetYesterdayDay(db, meterId, midday_yesterday_time)

        # Save the data to the database
        AddMinuteArray(db, meterId, new_daily_data)
        AddYesterdayMinuteArray(db, meterId, new_yesterday_daily_data)
        AddDayArraySummary(db, meterId, overall_data)


    # Extract the future data after the start time
    future_data = ExtractFutureVirtualMeterCsvData(constants.dataFilepath, start_time)

    # Start running the virtual meter from the start time
    if run_fast:
        RunFast(db, meterId, future_data, constants.addInterval)
    else:
        Run(db, meterId, future_data, start_time)

def StartFromNow(db, meterId, run_fast):
    """
    Starts the virtual meter from the current time. The function retrieves the current time, 
    rounded down to the nearest minute, and passes it to the StartFromPoint function to begin 
    running the virtual meter from this point.
    :param db: The reference to the Firebase database where meter data is stored.
    :param meterId: The unique identifier for the meter to be started from the current time.
    :param run_fast: A boolean flag that determines whether to use the fast simulation mode. If `True`, the `RunFast` function is used to add entries at a quicker pace based on the specified interval. If `False`, the standard `Run` function is used, simulating entries in real-time.
    :return: None
    """
    # Retrieve the current time and round it down to the nearest minute
    start_time = (datetime.now()).replace(second=0, microsecond=0)
    
    # Call StartFromPoint to start the virtual meter from the current time
    StartFromPoint(db, meterId, start_time, run_fast)

def Resume(db, meterId, run_fast):
    """
    Resumes the virtual meter from the most recent recorded time. If no previous data is found, it starts from the 
    current time. The function retrieves future data starting from the latest time and processes it.
    :param db: The reference to the database where the meter data is stored.
    :param meterId: The ID of the meter.
    :param run_fast: A boolean flag that determines whether to use the fast simulation mode. If `True`, the `RunFast` function is used to add entries at a quicker pace based on the specified interval. If `False`, the standard `Run` function is used, simulating entries in real-time.
    :return: None
    """

    # Get the latest recorded minute data from the database
    latest_time, _, _ = GetLatestMinute(db, meterId)
    
    # If there is no latest data, set the latest time to the current time (rounded down to the nearest minute)
    if latest_time is None:
        # Get current time as timestamp and then convert it to a datetime object
        latest_time_str = time.strftime('%Y-%m-%d %H:%M:00')
        latest_time = datetime.datetime.strptime(latest_time_str, '%Y-%m-%d %H:%M:%S')

    # Extract all future data starting from after the latest time
    future_data = ExtractFutureVirtualMeterCsvData(constants.dataFilepath, latest_time)
    
    # Start running the virtual meter from the latest time with the extracted future data
    if run_fast:
        RunFast(db, meterId, future_data, constants.addInterval)
    else:
        Run(db, meterId, future_data, future_data[0][0])

def Run(db, meterId, future_data, start_time):
    """
    This function processes future data entries and inserts them into the database at the appropriate times,
    adjusted by a start time offset to simulate running the meter at a different time.
    :param db: The reference to the Firebase database where the meter data is stored.
    :param meterId: The ID of the meter.
    :param future_data: The list of future data entries containing (entry_time, intensity, voltage).
    :param start_time: The datetime object representing the start time for the virtual meter.
    """
    # Calculate the offset between the actual current time and the desired start time
    current_time = datetime.datetime.now()
    start_time_offset = current_time - start_time

    print(f"Start Time: {start_time}")
    print(f"Current Time: {current_time}")
    print(f"Start Time Offset: {start_time_offset}")

    previous_entry_date = future_data[0][0].date()

    for entry in future_data:
        entry_time, intensity, voltage = entry

        # If it has moved on to the next day, calculate and save the overall data for the previous day
        if entry_time.date() > previous_entry_date:
            CalculateAndSaveOverallData(db, meterId)

        previous_entry_date = entry_time.date()

        # Get the current time at this point in the loop
        current_time = datetime.datetime.now()

        # Adjust the entry time by applying the start time offset (subtracting the offset)
        adjusted_entry_time = entry_time + start_time_offset

        # Add entry to database if the timepoint (with offset) has already passed
        if adjusted_entry_time < current_time:
            AddMinuteData(db, meterId, entry_time, intensity, voltage)

        # Calculate the time difference (in seconds) between the current time and the adjusted entry time
        time_to_wait = (adjusted_entry_time - current_time).total_seconds()
        print(f"Time to Wait: {time_to_wait} seconds for entry {entry_time} (adjusted: {adjusted_entry_time})")

        # If the entry time is in the future, wait until that time
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        # After waiting (or if already past), add the entry to the database
        AddMinuteData(db, meterId, entry_time, intensity, voltage)
        print(f"Adding entry to DB: {entry}")

    return

def RunFast(db, meterId, future_data, add_interval):
    """
    A simplified version of the Run function that adds entries at a faster pace based on the add_interval.
    To be used for demonstrations in the exhibition.
    :param db: The reference to the Firebase database.
    :param meterId: The ID of the meter.
    :param future_data: A list of future data entries containing (entry_time, intensity, voltage).
    :param add_interval: The interval in seconds between each data entry addition.
    """

    previous_entry_date = future_data[0][0].date()

    for entry in future_data:
        entry_time, intensity, voltage = entry

        # If it has moved on to the next day, calculate and save the overall data for the previous day
        if entry_time.date() > previous_entry_date:
            CalculateAndSaveOverallData(db, meterId)

        previous_entry_date = entry_time.date()

        # Add the entry to the database
        AddMinuteData(db, meterId, entry_time, intensity, voltage)
        print(f"Adding entry to DB: {entry}")

        # Wait for the specified interval before adding the next entry
        time.sleep(add_interval)

    return