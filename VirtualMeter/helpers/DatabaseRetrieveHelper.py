import datetime


def GetMinuteData(db, meterId: str, time: datetime) -> (float, float):
    """
    Gets the minute data for a specific time of day from dailyData table
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :param time: Time of day to the get the data from
    :return: Minute data values in form (Intensity, Voltage)
    """
    # Get the daily data for the meter
    doc_ref = db.collection("dailyData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        daily_data = doc.to_dict().get("seriesData", [])
        
        # Find the specific time entry in the seriesData
        for entry in daily_data:
            if entry["Date"] == time:
                return entry["Intensity"], entry["Voltage"]
    
    # If the data for the specified time isn't found, return None
    return None, None


def GetDaySummary(db, meterId: str, date: datetime) -> (float, float, float, float):
    """
    Gets the daily summary data from the overallData table given a specific date
    :param db: Database reference to get data from
    :param meterId: Document identifier representing meter to get data from
    :param date: Date to get data from
    :return: Daily summary data for date specified in the form (AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption)
    """
    # Get the overall data for the meter
    doc_ref = db.collection("overallData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        overall_data = doc.to_dict().get("data", [])
        
        # Find the summary for the specific date
        for entry in overall_data:
            if entry["Date"] == date:
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
        return doc.to_dict().get("seriesData", [])
    
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
        return doc.to_dict().get("seriesData", [])
    
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
