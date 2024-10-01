import constants
import datetime
from helpers.DatabaseMutationHelper import *
from helpers.DatabaseRetrieveHelper import *
from helpers.DatabaseUpdateHelper import *
from helpers.CsvDataHelper import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


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
    adds data up to and including the start time, will go to the next minute data and start running from there
    """
    latest_time, _, _ = GetLatestMinute(db, meterId)

    # Remove timezone awareness from latest_time and start_time if present
    if latest_time.tzinfo is not None:
        latest_time = latest_time.replace(tzinfo=None)
    
    if start_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=None)

    # If latest_time is greater than or equal to start_time, print error and return
    if latest_time >= start_time:
        print(f"Error: start_time {start_time} must be after the latest_time {latest_time}.")
        return

    missing_data_range = ExtractRangeVirtualMeterCsvData(constants.dataFilepath, latest_time, start_time)

    # same day
    if latest_time.date() == start_time.date():
        AddMinuteArray(db, meterId, missing_data_range)

    # start date is after one day after latest date
    elif abs((latest_time.date() - start_time.date()).days) <= 1:
        remaining_daily_data = []
        new_daily_data = []

        for entry in missing_data_range:
            entry_time = entry[0]

            # Ensure the entry time is timezone-naive
            if entry_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=None)

            # if entry has same date as the previous daily data
            if entry_time.date() == latest_time.date():
                remaining_daily_data.append(entry)
            # if the entry is the next date
            else:
                new_daily_data.append(entry)
        
        AddMinuteArray(db, meterId, remaining_daily_data) ## add the remaining daily data
        CalculateAndSaveOverallData(db, meterId) ## calculate summary from complete daily data and move daily data to yesterday series
        AddMinuteArray(db, meterId, new_daily_data) ## add new daily data

    ## more than a days difference between latest time and start time
    else:
        # get the daily data from the current day
        latest_daily_data = GetAllDaily(db, meterId) # the daily data from the latest time point
        historical_daily_data_dict = {}  # Dictionary to store daily data for all days between the latest time (including daily data already in the db) and day before start time

        historical_daily_data_dict[latest_time.date()] = latest_daily_data

        new_daily_data = []
        new_yesyerday_daily_data = []
        overall_data = []

        for entry in missing_data_range:
            entry_time = entry[0]

            # Ensure the entry time is timezone-naive
            if entry_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=None)

            # if entry is on the same date is the existing daily data table
            if entry_time.date() == latest_time.date():
                historical_daily_data_dict[latest_time.date()].append(entry)
            # if entry does not have the same date as latest time (existing daily data)
            else:
                if entry_time.date() not in historical_daily_data_dict:
                    historical_daily_data_dict[entry_time.date()] = []

                historical_daily_data_dict[entry_time.date()].append(entry)

                # if entry is the previous day to the start date
                if abs((latest_time.date() - entry_time.date()).days) <= 1:
                    new_yesyerday_daily_data.append(entry)
                
        # Calculate the daily summary for all other dates in the historical_daily_data_dict
        for date, date_data in historical_daily_data_dict.items():
            # Calculate the summary for the day using CalculateDaySummary
            day_summary = CalculateDaySummary(date_data)

            # Append the date and the corresponding summary to the overall_data list
            overall_data.append((date, day_summary))

    future_data = ExtractFutureVirtualMeterCsvData(constants.dataFilepath, start_time)

    Run(db, meterId, future_data)

def Resume(db, meterId):
    latest_time, _, _ = GetLatestMinute(db, meterId)
    
    if latest_time is None:
        latest_time = (datetime.now()).replace(second=0, microsecond=0)

    future_data = ExtractFutureVirtualMeterCsvData(constants.dataFilepath, latest_time)
    Run(db, meterId, future_data)

def Run(db , meterId, future_data):
    return