import datetime


def CalculateDaySummary(minuteIntervals: list[datetime, float, float]) -> (float, float, float, float):
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
