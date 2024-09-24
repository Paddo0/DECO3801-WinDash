from datetime import datetime, timedelta
from firebase_admin import firestore


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
        "yesterdayDate": datetime.now() - timedelta(days=1),
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
    Clears both the overallData collection and dailyData collection array data for a given meter
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
    :param db: Database reference to apply changes to
    :param meterId: Meter to clear daily data from
    :param time: Time of day to clear data after
    """
    # I recommend getting the date variable from the dailyData to avoid having to specify in time variable
    # Get the current daily data
    doc_ref = db.collection("dailyData").document(meterId)
    daily_data = doc_ref.get().to_dict()

    if daily_data is not None and "seriesData" in daily_data:
        series_data = daily_data["seriesData"]

        # Filter the entries based on the given time - NOT SURE IF I NEED TO CHECK EVERY ENTRY, MIGHT BE ABLE TO FIND FIRST ENTRY AND REMOVE ALL AFTER IT
        filtered_data = []
        for entry in series_data:
            if entry['Date'] <= time:
                filtered_data.append(entry)


        # Update the database with the filtered data
        doc_ref.update({"seriesData": filtered_data})

    return


def ClearOverallFromDate(db, meterId: str, date: datetime):
    """
    Given a date, removes all data summaries from overallData after that date
    :param db: Database reference to apply changes to
    :param meterId: Meter to clear daily summaries from
    :param date: Date of year to clear daily summaries after
    """
    # Get the current overall data
    doc_ref = db.collection("overallData").document(meterId)
    overall_data = doc_ref.get().to_dict()

    if overall_data is not None and "data" in overall_data:
        data = overall_data["data"]

        # Filter the summaries based on the given date - NOT SURE IF I NEED TO CHECK EVERY ENTRY, MIGHT BE ABLE TO FIND FIRST ENTRY AND REMOVE ALL AFTER IT
        filtered_data = []
        for entry in data:
            if entry['Date'] <= date:
                filtered_data.append(entry)


        # Update the database with the filtered data
        doc_ref.update({"data": filtered_data})

    return
