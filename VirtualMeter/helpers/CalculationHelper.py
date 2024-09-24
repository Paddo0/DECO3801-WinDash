import datetime


def CalculateDaySummary(minuteIntervals: list[datetime.datetime, float, float]) -> (float, float, float, float):
    """
    Given a list of minute interval data, calculate the daily summary to be saved into overallData.
    :param minuteIntervals: List representation of dailyData seriesData array in form (Date, Intensity, Voltage)
    :return: Summary of data in form (AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption)
    """
    # Initialise summary values
    total_intensity = 0
    max_intensity = float('-inf')
    min_intensity = float('inf')
    total_consumption = 0
    num_intervals = len(minuteIntervals)

    # Edge case: if the list is empty, return 0 for all values
    if num_intervals == 0:
        return 0, 0, 0, 0

    # Loop through the minute intervals and calculate summary values
    for entry in minuteIntervals:
        date, intensity, voltage = entry
        
        # Update total intensity
        total_intensity += intensity
        
        # Update max and min intensity
        if intensity > max_intensity:
            max_intensity = intensity
        if intensity < min_intensity:
            min_intensity = intensity
        
        # Update total consumption (intensity * voltage for each minute)
        total_consumption += intensity * voltage

    # Calculate average intensity
    average_intensity = total_intensity / num_intervals
    
    # Convert total consumption to kWh
    total_consumption_kWh = total_consumption / 1000 / 60  # Dividing by 1000 to convert watts to kilowatts, then by 60 to convert minutes to hours

    # Return the calculated values
    return average_intensity, max_intensity, min_intensity, total_consumption_kWh

def CalculateDaySummaryArray(days_data: list[list[tuple[datetime.datetime, float, float]]]) -> list[tuple[float, float, float, float]]:
    """
    Given a list of days, where each day contains minute interval data in the form (Date, Intensity, Voltage),
    calculates the daily summary for each day using the CalculateDaySummary function and returns a list of daily summaries.
    
    :param days_data: List of lists, where each sublist represents a day's data. 
                      Each sublist contains tuples in the form (Date, Intensity, Voltage).
    :return: List of daily summaries, where each summary is a tuple (AverageIntensity, MaxIntensity, MinIntensity, TotalConsumption).
    """
    summaries = []

    # Iterate through each day's data
    for day_data in days_data:
        # Use the existing CalculateDaySummary function for each day's data
        summary = CalculateDaySummary(day_data)

        # Append the calculated summary to the summaries list
        summaries.append(summary)

    return summaries
