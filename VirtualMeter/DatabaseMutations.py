

def AddMeter(db, meterId):
    """
    Adds a meter Id as a document with initial setup / structure with empty data arrays
    :param db: database reference to apply changes to
    :param meterId: variable to set as the document identifier
    """
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
    return


def ClearTodayData(db, meterId):
    """
    Removes all data from the dailyData seriesData saved
    :param db: database reference to apply changes to
    :param meterId: meter to remove today's data from
    """
    return


def AddMinuteArray(db, meterId, listOfData):
    """
    Adds a range of value to the dailyData seriesData array
    :param db: database reference to apply changes to
    :param meterId: meter to add series data to
    :param listOfData: List of minute data in form (time, intensity, voltage)
    """
    return


def SetCurrentDay(db, meterId, date):
    """
    Sets the value of currentDate within the dailyData collection
    :param db: database reference to apply changes to
    :param meterId: meter to set date for
    :param date: date to set
    """
    return


def AddDaySummary(db, meterId, date, summary):
    """
    Adds an entry to the overallData data array
    :param db: database reference to apply changes to
    :param meterId: meter to save data to
    :param date: date of the summary
    :param summary: data to be saved, in form (AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption)
    """
    return


def AddDayArraySummary(db, meterId, listOfData):
    """
    Adds a list of values to the overallData data array
    :param db: database reference to apply changes to
    :param meterId: meter to add list of data to
    :param listOfData: list of daily summary data in the form (date, summary)
    """
    return


def ClearOverallData(db, meterId):
    """
    Clears the data array in the overallData collection for the given meter
    :param db: database reference to apply changes to
    :param meterId: meter to clear the overall data for
    """
    return


def ClearAllData(db, meterId):
    """
    Clears both the overallData collection and dailyData collection array data for a given meter
    :param db: database reference to apply changes to
    :param meterId: meter to clear array data for
    """
    ClearTodayData(meterId)
    ClearOverallData(meterId)
    return

