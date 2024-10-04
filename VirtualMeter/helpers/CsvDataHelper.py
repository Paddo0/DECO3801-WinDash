import numpy as np
import sys
from helpers.DatabaseMutationHelper import *
import datetime


def GetData(filepath):
    # Loading in data
    data = np.loadtxt(filepath, dtype=type("e"), delimiter=';', skiprows=1)

    # Removing invalid rows
    data = data[data[:, 2] != "?"]
    return data


def LoadCsvData(db, meterId, filepath):
    # Getting csv data
    data = GetData(filepath)

    # Defining the final day to save to daily collection
    finalDay = datetime.datetime(2010, 11, 26)

    # Defining date offset to define dates recently (not from 2006)
    setDate = datetime.datetime(2024, 8, 30)
    timeDifference = setDate - finalDay

    # Iterating over array to process data
    overallData = []
    dailyData = []
    prevDate = None
    numberIntervals = 0
    averageIntensity = 0
    maxIntensity = 0
    minIntensity = sys.float_info.max
    totalConsumption = 0
    for row in data:
        # Converting values
        dateString, timeString = row[0:2]
        date = datetime.datetime.strptime(dateString, '%d/%m/%Y')
        (voltage, intensity) = row[4:6]
        voltage = float(voltage)
        intensity = float(intensity)

        if date == finalDay:
            # Daily data
            time = datetime.datetime.strptime(dateString + timeString, '%d/%m/%Y%H:%M:%S')
            dailyData.append([
                time + timeDifference,
                intensity,
                voltage,
            ])

        else:
            # Overall data
            # If part of the same day, calculate the summary for the day, else, save the summary and reset values
            if date == prevDate:
                # Calculate Summary
                numberIntervals += 1
                averageIntensity += intensity
                maxIntensity = max(maxIntensity, intensity)
                minIntensity = min(minIntensity, intensity)
                totalConsumption += intensity * voltage

            elif prevDate is not None:
                # Saving data
                print("Saving", date)
                totalConsumption = totalConsumption / 1000 / 1440
                overallData.append([
                    date + timeDifference,  # Date
                    (averageIntensity / numberIntervals,  # AverageIntensity
                     maxIntensity,  # MaximumIntensity
                     minIntensity,  # MinimumIntensity
                     totalConsumption)  # TotalConsumption
                ])

                # Resetting summary values
                numberIntervals = 1
                averageIntensity = intensity
                maxIntensity = intensity
                minIntensity = intensity
                totalConsumption = intensity

            # Setting previous date
            prevDate = date

    # Saving dates to database
    AddMinuteArray(db, meterId, dailyData)
    AddDayArraySummary(db, meterId, overallData)

    return

## Below function is not used - using ExtractPastVirtualMeterCsvData, ExtractFutureVirtualMeterCsvData and ExtractRangeVirtualMeterCsvData instead.
def ExtractAllVirtualMeterCsvData(filepath, start_time):
    """
    Extracts all virtual meter data from a CSV file, processes it, and organizes it into specific groups:
    - Data for the day before the start date
    - Data for the start date itself, split into minute intervals up to the start time (does not include data after start time)
    - Summarized data for all days before the start date
    - Data for all days and times after the start time
    The function applies a time shift of 14 years to align the historical dates in the CSV with current dates.
    :param filepath: The path to the CSV file containing the virtual meter data
    :param start_time: A `datetime` object representing the exact time from which the data processing should start
    :return: A tuple containing:
             - yesterday_daily_data: A list of tuples (time, intensity, voltage) for the day before the start date
             - daily_data: A list of tuples (time, intensity, voltage) for the start date up to the start time
             - overall_data: A list of daily summaries for all days before the start date
             - future_daily_data: A list of tuples (time, intensity, voltage) for all times after the start time
    """
    
    # Calculate the date for the day before the start date
    day_before_start_time = start_time.date() - datetime.timedelta(days=1)

    # Load the CSV data from the given filepath
    data = GetData(filepath)

    # Define a time difference of 14 years to shift historical dates to align with the current year
    date_end_date = datetime.datetime(2010, 11, 26)  # Historical end date in the CSV
    virtual_end_date = datetime.datetime(2024, 11, 26)  # Current virtual meter end date
    time_difference = virtual_end_date - date_end_date  # Time offset of 14 years

    # Initialise data structures
    overall_data = []  # List to store summaries of all days except the start day and day before
    yesterday_daily_data = []  # List to store minute-interval data for the day before the start date
    daily_data = []  # List to store minute-interval data for the start date to be uploaded directly to the DailyData table
    future_daily_data = []  # List to store minute-interval data for all data after the start date/time
    historical_daily_data_dict = {}  # Dictionary to store daily data for all days before the start date

    # Loop through each row in the CSV data
    for row in data:
        # Extract the date and time from the CSV row
        date_string, time_string = row[0:2]
        # Combine the date and time, and apply the time difference offset to align it with the current year
        time = datetime.datetime.strptime(date_string + time_string, '%d/%m/%Y%H:%M:%S') + time_difference
        # Extract the date part from the datetime object
        date = time.date()

        # Extract voltage and intensity data from the CSV row and ensure they are floats
        voltage, intensity = float(row[4]), float(row[5])

        # Process data based on the date
        if date == day_before_start_time:
            # If the date is the day before the start date, append the data to yesterday_daily_data
            yesterday_daily_data.append((time, intensity, voltage))
        elif date == start_time.date():
            if time <= start_time:
                # If the time is before or at the start time, add data to daily_data
                daily_data.append((time, intensity, voltage))
            else:
                # If the time is after the start time, add data to future_daily_data
                future_daily_data.append((time, intensity, voltage))
        else:
            if time < start_time:
                # For times before the start date, store data in historical_daily_data_dict
                if date not in historical_daily_data_dict:
                    historical_daily_data_dict[date] = []
                historical_daily_data_dict[date].append((time, intensity, voltage))
            else:
                # For times after the start time, add data to future_daily_data
                future_daily_data.append((time, intensity, voltage))

    # Calculate the daily summary for all other dates in the historical_daily_data_dict
    for date, date_data in historical_daily_data_dict.items():
        # Calculate the summary for the day using CalculateDaySummary
        day_summary = CalculateDaySummary(date_data)
        
        # Append the date and the corresponding summary to the overall_data list
        overall_data.append((date, day_summary))

    # Return the collected data
    return yesterday_daily_data, daily_data, overall_data, future_daily_data

def ExtractPastVirtualMeterCsvData(filepath, start_time):
    """
    Extracts all past virtual meter data from a CSV file, processes it, and organizes it into specific groups:
    - Data for the day before the start date
    - Data for the start date itself, split into minute intervals up to the start time (does not include data after start time)
    - Summarized data for all days before the start date
    The function applies a time shift of 14 years to align the historical dates in the CSV with current dates.
    :param filepath: The path to the CSV file containing the virtual meter data
    :param start_time: A `datetime` object representing the exact time from which the data processing should start (inclusive)
    :return: A tuple containing:
             - yesterday_daily_data: A list of tuples (time, intensity, voltage) for the day before the start date
             - daily_data: A list of tuples (time, intensity, voltage) for the start date up to the start time
             - overall_data: A list of daily summaries for all days before the start date
    """
    
    # Calculate the date for the day before the start date
    day_before_start_time = start_time.date() - datetime.timedelta(days=1)

    # Load the CSV data from the given filepath
    data = GetData(filepath)

    # Define a time difference of 14 years to shift historical dates to align with the current year
    date_end_date = datetime.datetime(2010, 11, 26)  # Historical end date in the CSV
    virtual_end_date = datetime.datetime(2024, 11, 26)  # Current virtual meter end date
    time_difference = virtual_end_date - date_end_date  # Time offset of 14 years

    # Initialise data structures
    overall_data = []  # List to store summaries of all days except the start day and day before
    yesterday_daily_data = []  # List to store minute-interval data for the day before the start date
    daily_data = []  # List to store minute-interval data for the start date to be uploaded directly to the DailyData table
    historical_daily_data_dict = {}  # Dictionary to store daily data for all days before the start date

    # Loop through each row in the CSV data
    for row in data:
        # Extract the date and time from the CSV row
        date_string, time_string = row[0:2]
        # Combine the date and time, and apply the time difference offset to align it with the current year
        time = datetime.datetime.strptime(date_string + time_string, '%d/%m/%Y%H:%M:%S') + time_difference
        # Extract the date part from the datetime object
        date = time.date()

        # Extract voltage and intensity data from the CSV row and ensure they are floats
        voltage, intensity = float(row[4]), float(row[5])

        # Process data based on the date
        if date == day_before_start_time:
            # If the date is the day before the start date, append the data to yesterday_daily_data
            yesterday_daily_data.append((time, intensity, voltage))
            if date not in historical_daily_data_dict:
                    historical_daily_data_dict[date] = []
            historical_daily_data_dict[date].append((time, intensity, voltage))

        elif date == start_time.date():
            if time <= start_time:
                # If the time is before or at the start time, add data to daily_data
                daily_data.append((time, intensity, voltage))
                
        else:
            if time < start_time:
                # For times before the start date, store data in historical_daily_data_dict
                if date not in historical_daily_data_dict:
                    historical_daily_data_dict[date] = []
                historical_daily_data_dict[date].append((time, intensity, voltage))

    # Calculate the daily summary for all other dates in the historical_daily_data_dict
    for date, date_data in historical_daily_data_dict.items():
        # Calculate the summary for the day using CalculateDaySummary
        day_summary = CalculateDaySummary(date_data)
        
        # calculates the midday date for the specified date
        midday_date = date_data[0][0].replace(hour=12, minute=0, second=0)
        # Append the date and the corresponding summary to the overall_data list
        overall_data.append((midday_date, day_summary))

    # Return the collected data
    return yesterday_daily_data, daily_data, overall_data

def ExtractFutureVirtualMeterCsvData(filepath, start_time):
    """
    Extracts virtual meter data from a CSV file and collects all data points that occur after the specified start time.
    The function applies a time shift of 14 years to align the historical dates in the CSV with current dates.
    :param filepath: The path to the CSV file containing the virtual meter data.
    :param start_time: A `datetime` object representing the cutoff time. Only data points after this time will be collected (exclusive).
    :return: A list of tuples containing (time, intensity, voltage) for all times after the specified start_time.
    """

    # Load the CSV data from the given filepath
    data = GetData(filepath)

    # Define a time difference of 14 years to shift historical dates to align with the current year
    date_end_date = datetime.datetime(2010, 11, 26)  # Historical end date in the CSV
    virtual_end_date = datetime.datetime(2024, 11, 26)  # Current virtual meter end date
    time_difference = virtual_end_date - date_end_date  # Time offset of 14 years

    # Initialise a list to store minute-interval data for all times after the start_time
    future_daily_data = []

    # Ensure the start_time is timezone-naive
    if start_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=None)

    # Loop through each row in the CSV data
    for row in data:
        # Extract the date and time from the CSV row
        date_string, time_string = row[0:2]
        # Combine date and time, apply the time difference offset to align with the current year
        time = datetime.datetime.strptime(date_string + time_string, '%d/%m/%Y%H:%M:%S') + time_difference

        # Extract voltage and intensity data from the CSV row and ensure they are numeric
        voltage, intensity = float(row[4]), float(row[5])

        # Ensure the time is timezone-naive
        if time.tzinfo is not None:
            time = time.replace(tzinfo=None)

        # Check if the current time is after the specified start_time
        if time > start_time:
            # Append the data (time, intensity, voltage) to the future_daily_data list
            future_daily_data.append((time, intensity, voltage))

    # Return the collected data
    return future_daily_data

def ExtractRangeVirtualMeterCsvData(filepath, start_time, end_time):
    """
    Extracts virtual meter data from a CSV file and collects all data points that occur between two specified time points.
    The function applies a time shift of 14 years to align the historical dates with the current year.
    :param filepath: The path to the CSV file containing the virtual meter data.
    :param start_time: A `datetime` object representing the start of the time range (exclusive).
    :param end_time: A `datetime` object representing the end of the time range (inclusive).
    :return: A list of tuples containing (time, intensity, voltage) for all times between the specified start_time and end_time.
    """

    # Load the CSV data from the given filepath
    data = GetData(filepath)

    # Define a time difference of 14 years to shift historical dates to align with the current year
    date_end_date = datetime.datetime(2010, 11, 26)  # Historical end date in the CSV
    virtual_end_date = datetime.datetime(2024, 11, 26)  # Current virtual meter end date
    time_difference = virtual_end_date - date_end_date  # Time offset of 14 years

    # Initialise a list to store minute-interval data for the start time to the end time
    minute_data = []

    # Loop through each row in the CSV data
    for row in data:
        # Extract the date and time from the CSV row
        date_string, time_string = row[0:2]
        # Combine the date and time, apply the time difference offset to align with the current year
        time = datetime.datetime.strptime(date_string + time_string, '%d/%m/%Y%H:%M:%S') + time_difference

        # Extract voltage and intensity data from the CSV row and ensure they are numeric
        voltage, intensity = float(row[4]), float(row[5])

        # Check if the current time falls within the specified start_time and end_time range
        if start_time < time <= end_time:
            # Append the data (time, intensity, voltage) to the minute_data list
            minute_data.append((time, intensity, voltage))

    # Return the collected minute-interval data
    return minute_data
