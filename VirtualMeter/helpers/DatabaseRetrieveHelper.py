import datetime


def GetMinuteData(db, meterId: str, time: datetime) -> (float, float):
    """
    Gets the minute data for a specific time of day from dailyData table
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :param time: Time of day to the get the data from
    :return: Minute data values in form (Intensity, Voltage)
    """
    return


def GetDaySummary(db, meterId: str, date: datetime) -> (float, float, float, float):
    """
    Gets the daily summary data from the overallData table given a specific date
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :param date: Date to get data from
    :return: Daily summary data for date specified in the form (AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption)
    """
    return


def GetLatestMinute(db, meterId: str) -> (datetime, float, float):
    """
    Gets the most recent entry in the dailyData table
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: Most recent minute data in database
    """
    return


def GetLatestDaySummary(db, meterId: str) -> (datetime, float, float, float, float):
    """
    Gets the most recent entry in the overallData table
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: Most recent daily summary data in database
    """
    return


def GetAllDaily(db, meterId: str) -> (list[datetime, float, float]):
    """
    Gets a list of all the dailyData seriesData array for a specified meter
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: List of dailyData seriesData array in the form List(Date, Intensity, Voltage)
    """
    return


def GetAllOverall(db, meterId: str) -> (list[datetime, float, float, float, float]):
    """
    Gets a list of all the overallData data array for a specific meter
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: List of overallData data array in form List(Date, AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption)
    """
    return
