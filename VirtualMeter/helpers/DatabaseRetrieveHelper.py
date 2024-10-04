import datetime
from .DatabaseMutationHelper import *

def GetMinuteData(db, meterId: str, time: datetime) -> (float, float):
    """
    Gets the minute data for a specific time of day from the dailyData table.
    Ensures that both 'time' and 'Date' entries are timezone-naive for comparison.
    :param db: Database reference to get data from.
    :param meterId: Document identifier representing meter to get data from.
    :param time: Time of day to get the data from (should be timezone-naive).
    :return: Minute data values in the form (Intensity, Voltage).
    """
    # Ensure that the 'time' passed is timezone-naive
    if time.tzinfo is not None:
        time = time.replace(tzinfo=None)

    # Get the daily data for the meter
    doc_ref = db.collection("dailyData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        daily_data = doc.to_dict().get("seriesData", [])
        
        # Find the specific time entry in the seriesData
        for entry in daily_data:
            entry_time = entry["Date"]

            # Ensure the entry time is timezone-naive
            if entry_time.tzinfo is not None:
                entry_time = entry_time.replace(tzinfo=None)

            # Compare the timezone-naive datetime objects
            if entry_time == time:
                return entry["Intensity"], entry["Voltage"]
    
    # If the data for the specified time isn't found, return None
    return None, None



def GetDaySummary(db, meterId: str, date: datetime) -> (float, float, float, float):
    """
    Gets the daily summary data from the overallData table given a specific date.
    Ensures both the input 'date' and 'Date' entries from Firestore are timezone-naive.
    
    :param db: Database reference to get data from.
    :param meterId: Document identifier representing the meter to get data from.
    :param date: Date to get data from (should be timezone-naive).
    :return: Daily summary data for the date specified in the form (AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption).
    """
    # Ensure the input 'date' is timezone-naive
    if date.tzinfo is not None:
        date = date.replace(tzinfo=None)

    # Get the overall data for the meter
    doc_ref = db.collection("overallData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        overall_data = doc.to_dict().get("data", [])
        
        # Find the summary for the specific date
        for entry in overall_data:
            entry_date = entry["Date"]

            # Ensure the entry 'Date' is timezone-naive
            if entry_date.tzinfo is not None:
                entry_date = entry_date.replace(tzinfo=None)

            # Compare timezone-naive datetime objects
            if entry_date == date:
                return (entry["AverageIntensity"], entry["MaximumIntensity"], 
                        entry["MinimumIntensity"], entry["TotalConsumption"])
    
    # If the summary for the specified date isn't found, return None
    return None, None, None, None



def GetLatestMinute(db, meterId: str) -> (datetime, float, float):
    """
    Gets the most recent entry in the dailyData table
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: Most recent minute data in database
    """
    # Get the daily data for the meter
    doc_ref = db.collection("dailyData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        daily_data = doc.to_dict().get("seriesData", [])
        
        # Assuming the most recent entry is the last one in the list
        if daily_data:
            latest_entry = daily_data[-1]
            return latest_entry["Date"], latest_entry["Intensity"], latest_entry["Voltage"]
    
    # If no data is found, return None
    return None, None, None


def GetLatestDaySummary(db, meterId: str) -> (datetime, float, float, float, float):
    """
    Gets the most recent entry in the overallData table
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: Most recent daily summary data in database
    """
    # Get the overall data for the meter
    doc_ref = db.collection("overallData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        overall_data = doc.to_dict().get("data", [])
        
        # Assuming the most recent entry is the last one in the list
        if overall_data:
            latest_entry = overall_data[-1]
            return (latest_entry["Date"], latest_entry["AverageIntensity"], 
                    latest_entry["MaximumIntensity"], latest_entry["MinimumIntensity"], 
                    latest_entry["TotalConsumption"])
    
    # If no data is found, return None
    return None, None, None, None, None


def GetAllDaily(db, meterId: str) -> (list[datetime, float, float]):
    """
    Gets a list of all the dailyData seriesData array for a specified meter
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: List of dailyData seriesData array in the form List(Date, Intensity, Voltage)
    """
    # Get the daily data for the meter
    doc_ref = db.collection("dailyData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        daily_data = doc.to_dict().get("seriesData", [])
        transformed_Data = transform_series_data(daily_data)
        return transformed_Data

    # If no data is found, return an empty list
    return []

def GetAllYesterdayDailyData(db, meterId: str) -> (list[datetime, float, float]):
    """
    Gets a list of all the previousDayDailyData seriesData array for a specified meter
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: List of previousDayDailyData seriesData array in the form List(Date, Intensity, Voltage)
    """
    # Get the daily data for the meter
    doc_ref = db.collection("previousDayDailyData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        yesterday_daily_data = doc.to_dict().get("seriesData", [])
        transformed_Data = transform_series_data(yesterday_daily_data)
        return transformed_Data
    
    # If no data is found, return an empty list
    return []


def GetAllOverall(db, meterId: str) -> (list[datetime, float, float, float, float]):
    """
    Gets a list of all the overallData data array for a specific meter
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :return: List of overallData data array in form List(Date, AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption)
    """
    # Get the overall data for the meter
    doc_ref = db.collection("overallData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        return doc.to_dict().get("data", [])
    
    # If no data is found, return an empty list
    return []
