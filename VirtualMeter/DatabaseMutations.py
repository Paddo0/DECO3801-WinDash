import datetime
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

    # Setting data
    db.collection("dailyData").document(meterId).set(dailySetup)
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
    ClearOverallData(db, meterId)

    return

