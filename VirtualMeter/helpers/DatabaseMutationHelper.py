from datetime import datetime, timedelta
from firebase_admin import firestore
from helpers.CalculationHelper import *


def AddMeter(db, meterId):
    """
    Adds a meter Id as a document with initial setup / structure with empty data arrays
    :param db: database reference to apply changes to
    :param meterId: variable to set as the document identifier
    """
    # Default dailyData collection structure
    dailySetup = {
        "currentDate": datetime.datetime.now(),
        "seriesData": [],
    }

    # Default overallData collection structure
    overallSetup = {
        "data": [],
    }

    previousDayDailyDataSetup = {
        "yesterdayDate": datetime.datetime.now() - timedelta(days=1),
        "seriesData": [],
    }

    # Setting data
    db.collection("dailyData").document(meterId).set(dailySetup)
    db.collection("previousDayDailyData").document(meterId).set(previousDayDailyDataSetup)
    db.collection("overallData").document(meterId).set(overallSetup)

    return


def AddMinuteData(db, meterId, time, intensity, voltage):
    """
    Adds an entry to the daily series data
    :param db: database reference to apply changes to
    :param meterId: meter to add data to
    :param time: time of observation
    :param intensity: average amps used over interval
    :param voltage: average voltage over interval
    """

    # Defining array entry
    seriesDataEntry = {
        "Date": time,
        "Intensity": intensity,
        "Voltage": voltage,
    }

    # Adding entry to database
    db.collection("dailyData").document(meterId).update({"seriesData": firestore.ArrayUnion([seriesDataEntry])})

    return


def ClearTodayData(db, meterId):
    """
    Removes all data from the dailyData seriesData saved
    :param db: database reference to apply changes to
    :param meterId: meter to remove today's data from
    """

    # Setting seriesData to empty array
    db.collection("dailyData").document(meterId).update({"seriesData": []})

    return

def ClearYesterdayData(db, meterId):
    """
    Removes all data from the previousDayDailyData seriesData saved
    :param db: database reference to apply changes to
    :param meterId: meter to remove today's data from
    """

    # Setting seriesData to empty array
    db.collection("previousDayDailyData").document(meterId).update({"seriesData": []})

    return

def AddMinuteArray(db, meterId, listOfData):
    """
    Adds a range of value to the dailyData seriesData array
    :param db: database reference to apply changes to
    :param meterId: meter to add series data to
    :param listOfData: List of minute data in form (time, intensity, voltage)
    """

    # Unpacking list into entry format
    listOfEntries = []
    for (time, intensity, voltage) in listOfData:
        listOfEntries.append({
            "Date": time,
            "Intensity": intensity,
            "Voltage": voltage,
        })

    # Adding entries to database
    db.collection("dailyData").document(meterId).update({"seriesData": firestore.ArrayUnion(listOfEntries)})

    return

def AddYesterdayMinuteArray(db, meterId, listOfData):
    """
    Adds a range of value to the previousDayDailyData seriesData array
    :param db: database reference to apply changes to
    :param meterId: meter to add series data to
    :param listOfData: List of minute data in form (time, intensity, voltage)
    """

    # Unpacking list into entry format
    listOfEntries = []
    for (time, intensity, voltage) in listOfData:
        listOfEntries.append({
            "Date": time,
            "Intensity": intensity,
            "Voltage": voltage,
        })

    # Adding entries to database
    db.collection("previousDayDailyData").document(meterId).update({"seriesData": firestore.ArrayUnion(listOfEntries)})

    return


def SetCurrentDay(db, meterId, date):
    """
    Sets the value of currentDate within the dailyData collection
    :param db: database reference to apply changes to
    :param meterId: meter to set date for
    :param date: date to set
    """

    # Updating value in database
    db.collection("dailyData").document(meterId).update({"currentDate": date})

    return

def SetYesterdayDay(db, meterId, date):
    """
    Sets the value of yesterdayDate within the previousDayDailyData collection
    :param db: database reference to apply changes to
    :param meterId: meter to set date for
    :param date: date to set
    """

    # Updating value in database
    db.collection("previousDayDailyData").document(meterId).update({"yesterdayDate": date})

    return


def AddDaySummary(db, meterId, date, summary):
    """
    Adds an entry to the overallData data array
    :param db: database reference to apply changes to
    :param meterId: meter to save data to
    :param date: date of the summary
    :param summary: data to be saved, in form (AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption)
    """

    # Unpacking variables
    (averageIntensity, maxIntensity, minIntensity, totalConsumption) = summary

    # Defining array entry
    daySummaryEntry = {
        "AverageIntensity": averageIntensity,
        "Date": date,
        "MaximumIntensity": maxIntensity,
        "MinimumIntensity": minIntensity,
        'TotalConsumption': totalConsumption,
    }

    # Adding entry to database
    db.collection("overallData").document(meterId).update({"data": firestore.ArrayUnion([daySummaryEntry])})

    return


def AddDayArraySummary(db, meterId, listOfData):
    """
    Adds a list of values to the overallData data array
    :param db: database reference to apply changes to
    :param meterId: meter to add list of data to
    :param listOfData: list of daily summary data in the form (date, summary)
    """

    # Unpacking list into entry format
    listOfEntries = []
    for date, summary in listOfData:
        # Unpacking summary
        (averageIntensity, maxIntensity, minIntensity, totalConsumption) = summary

        # Adding entry
        listOfEntries.append({
            "AverageIntensity": averageIntensity,
            "Date": date,
            "MaximumIntensity": maxIntensity,
            "MinimumIntensity": minIntensity,
            'TotalConsumption': totalConsumption,
        })

    # Adding entries to database
    db.collection("overallData").document(meterId).update({"data": firestore.ArrayUnion(listOfEntries)})

    return


def ClearOverallData(db, meterId):
    """
    Clears the data array in the overallData collection for the given meter
    :param db: database reference to apply changes to
    :param meterId: meter to clear the overall data for
    """

    # Setting data to empty array
    db.collection("overallData").document(meterId).update({"data": []})

    return


def ClearAllData(db, meterId):
    """
    Clears both the overallData collection, previousDayDailyData collection and dailyData collection array data for a given meter
    :param db: database reference to apply changes to
    :param meterId: meter to clear array data for
    """

    ClearTodayData(db, meterId)
    ClearYesterdayData(db, meterId)
    ClearOverallData(db, meterId)

    return


def ClearDailyFromTime(db, meterId: str, time: datetime):
    """
    Given a time of day, clears all data from dailyData after that point
    Combines the date from the currentDate in dailyData with the time provided in the 'time' argument
    
    :param db: Database reference to apply changes to
    :param meterId: Meter to clear daily data from
    :param time: Time of day (hour, minute, second) to clear data after
    """
    # Ensure that the 'time' passed is timezone-naive
    if time.tzinfo is not None:
        time = time.replace(tzinfo=None)

    # Get the current daily data
    doc_ref = db.collection("dailyData").document(meterId)
    daily_data = doc_ref.get().to_dict()

    if daily_data is not None and "seriesData" in daily_data and "currentDate" in daily_data:
        series_data = daily_data["seriesData"]
        current_date = daily_data["currentDate"]

        # Ensure the current_date is timezone-naive
        if current_date.tzinfo is not None:
            current_date = current_date.replace(tzinfo=None)

        # Combine the current date with the specified time to create a new datetime object
        combined_datetime = datetime.datetime(
            current_date.year, current_date.month, current_date.day,
            time.hour, time.minute, time.second
        )

        # Filter the entries based on the combined datetime
        filtered_data = []
        for entry in series_data:
            entry_time = entry['Date']

            # Ensure the entry time is timezone-naive
            if entry_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=None)

            # Compare timezone-naive datetime objects
            if entry_time <= combined_datetime:
                filtered_data.append(entry)

        # Update the database with the filtered data
        doc_ref.update({"seriesData": filtered_data})

    else:
        print(f"Data not found or 'currentDate' missing for meterId {meterId}")

    return

def ClearOverallFromDate(db, meterId: str, date: datetime):
    """
    Given a date, removes all data summaries from overallData after that date
    :param db: Database reference to apply changes to
    :param meterId: Meter to clear daily summaries from
    :param date: Date of year to clear daily summaries after
    """
    # Ensure that the 'date' passed is timezone-naive
    if date.tzinfo is not None:
        date = date.replace(tzinfo=None)

    # Get the current overall data
    doc_ref = db.collection("overallData").document(meterId)
    overall_data = doc_ref.get().to_dict()

    if overall_data is not None and "data" in overall_data:
        data = overall_data["data"]

        # Filter the summaries based on the given date
        filtered_data = []
        for entry in data:
            entry_date = entry['Date']

            # Ensure the entry date is timezone-naive
            if entry_date.tzinfo is not None:
                entry_date = entry_date.replace(tzinfo=None)

            # Compare timezone-naive datetime objects
            if entry_date <= date:
                filtered_data.append(entry)

        # Update the database with the filtered data
        doc_ref.update({"data": filtered_data})

    return

def MoveDailyDataToPreviousDay(db, meterId: str):
    """
    Moves the current day's data from the dailyData collection to the previousDayDailyData collection and clears dailyData
    :param db: Database reference to apply changes to
    :param meterId: Meter ID to identify which document to move
    """
    # Get the current day's data from the dailyData collection
    doc_ref_daily = db.collection("dailyData").document(meterId)
    daily_data_doc = doc_ref_daily.get()

    if daily_data_doc.exists:
        daily_data = daily_data_doc.to_dict()

        # Move the daily data to the previousDayDailyData collection
        doc_ref_previous_day = db.collection("previousDayDailyData").document(meterId)
        
        # Update the previous day's data with the current daily data (overwrite)
        doc_ref_previous_day.set({
            'yesterdayDate': daily_data.get('currentDate', datetime.datetime.now()),
            'seriesData': daily_data.get('seriesData', [])
        })

        # Clear the current day's data in the dailyData collection
        doc_ref_daily.update({
            'seriesData': [],
            'currentDate': datetime.datetime.now()  # Reset to today's date
        })

        print(f"Moved daily data to previousDayDailyData for meterId {meterId} and cleared dailyData.")
    else:
        print(f"No daily data found for meterId {meterId}.")

    return

def CalculateAndSaveOverallData(db, meterId: str):
    """
    Calculates the daily summary from the dailyData collection, saves it to overallData using AddDaySummary and then moves the daily data to previousDayDailyData.
    :param db: Database reference to apply changes to
    :param meterId: Meter ID to identify which document to calculate and save
    """
    # Get the current day's data from the dailyData collection
    doc_ref_daily = db.collection("dailyData").document(meterId)
    daily_data_doc = doc_ref_daily.get()

    if daily_data_doc.exists:
        daily_data = daily_data_doc.to_dict()
        series_data = daily_data.get('seriesData', [])

        if series_data:
            transformed_series_data = transform_series_data(series_data)
            # Call CalculateDaySummary to get the daily summary from minute-level data
            average_intensity, max_intensity, min_intensity, total_consumption_kWh = CalculateDaySummary(transformed_series_data)

            # Get the current date from the daily data, or use today's date as a fallback
            current_date = daily_data.get('currentDate', datetime.datetime.now())

            # Use AddDaySummary to save the daily summary to the overallData collection
            AddDaySummary(db, meterId, current_date, (average_intensity, max_intensity, min_intensity, total_consumption_kWh))

            print(f"Saved daily summary for {meterId} to overallData on {current_date}.")

            # Move current day's data to previousDayDailyData
            MoveDailyDataToPreviousDay(db, meterId)

        else:
            print(f"No series data found for {meterId}.")
    else:
        print(f"No daily data found for {meterId}.")

    return

def transform_series_data(series_data):
    """
    Transforms the series_data from the Firestore format into a list of (datetime, float, float) tuples.
    :param series_data: List of dictionaries retrieved from Firestore with 'Date', 'Intensity', and 'Voltage' keys.
    :return: List of tuples in the form (datetime, intensity, voltage).
    """
    transformed_data = []
    
    for entry in series_data:
        # Extract and ensure 'Date', 'Intensity', and 'Voltage' are in the right format
        date = entry["Date"]  # This is already a datetime object
        intensity = float(entry["Intensity"])  # Convert intensity to float
        voltage = float(entry["Voltage"])      # Convert voltage to float
        
        # Append the tuple (datetime, intensity, voltage) to the transformed list
        transformed_data.append((date, intensity, voltage))
    
    return transformed_data