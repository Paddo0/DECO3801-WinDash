import datetime


def UpdateDailyDataDifference(db, meterId: str, timeDifference: datetime):
    """
    Shifts the date of every date data in the dailyData table (including currentDate)
    :param db: Database reference to update data for
    :param meterId: Document identifier representing meter to update data for
    :param timeDifference: The difference in time to shift each of the entries by
    """
    return


def UpdateMonthlyDataDifference(db, meterId: str, dateDifference: datetime):
    """
    Shifts the date of every date data within the overallData table
    :param db: Database reference to update data for
    :param meterId: Document identifier representing meter to update data for
    :param dateDifference: The difference in time to shift each of the entries by
    """
    return


def UpdateAllDatetimeData(db, meterId: str, currentDate: datetime):
    """
    Sets the date data for all tables around a specified point, the given time represents the start of the day to
    pivot the data about; i.e. the overallData's latest entry will be the day before the date given, and the dailyData
    table will start the current seriesData from the given time
    :param db: Database reference to update data for
    :param meterId: Document identifier representing meter to update data for
    :param currentDate: The start of the day to set a meter's date data to, ensure this date is set to midnight at the
                        start of the day to act as a starting point for the dailyData seriesData
    """
    # This is the function to actually calculate the time difference, make sure the difference calculated for the
    # overallData has the most recent entry as the day before the current date and for the currentDate variable to be
    # based on midnight for the user. We may need to have a discussion based on the start of days within the database
    # since firebase stores it as UTC+10 which is AEST which should be fine but is something that should be kept in
    # mind
    return
