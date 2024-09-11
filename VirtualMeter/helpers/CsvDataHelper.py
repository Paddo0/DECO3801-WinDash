import numpy as np
import sys
from VirtualMeter.helpers.DatabaseMutationHelper import *


def GetData(filepath):
    # Loading in data
    data = np.loadtxt(filepath, dtype=type("e"), delimiter=';', skiprows=1)

    # Removing invalid rows
    data = data[data[:, 2] != "?"]
    return data


def LoadCsvData(db, meterId, filepath):
    # Getting csv data
    data = GetData(filepath)

    # Defining the final day to save to daily collection
    finalDay = datetime.datetime(2010, 11, 26)

    # Defining date offset to define dates recently (not from 2006)
    setDate = datetime.datetime(2024, 8, 30)
    timeDifference = setDate - finalDay

    # Iterating over array to process data
    overallData = []
    dailyData = []
    prevDate = None
    numberIntervals = 0
    averageIntensity = 0
    maxIntensity = 0
    minIntensity = sys.float_info.max
    totalConsumption = 0
    for row in data:
        # Converting values
        dateString, timeString = row[0:2]
        date = datetime.datetime.strptime(dateString, '%d/%m/%Y')
        (voltage, intensity) = row[4:6]
        voltage = float(voltage)
        intensity = float(intensity)

        if date == finalDay:
            # Daily data
            time = datetime.datetime.strptime(dateString + timeString, '%d/%m/%Y%H:%M:%S')
            dailyData.append([
                time + timeDifference,
                intensity,
                voltage,
            ])

        else:
            # Overall data
            # If part of the same day, calculate the summary for the day, else, save the summary and reset values
            if date == prevDate:
                # Calculate Summary
                numberIntervals += 1
                averageIntensity += intensity
                maxIntensity = max(maxIntensity, intensity)
                minIntensity = min(minIntensity, intensity)
                totalConsumption += intensity * voltage

            elif prevDate is not None:
                # Saving data
                print("Saving", date)
                totalConsumption = totalConsumption / 1000 / 1440
                overallData.append([
                    date + timeDifference,  # Date
                    (averageIntensity / numberIntervals,  # AverageIntensity
                     maxIntensity,  # MaximumIntensity
                     minIntensity,  # MinimumIntensity
                     totalConsumption)  # TotalConsumption
                ])

                # Resetting summary values
                numberIntervals = 1
                averageIntensity = intensity
                maxIntensity = intensity
                minIntensity = intensity
                totalConsumption = intensity

            # Setting previous date
            prevDate = date

    # Saving dates to database
    AddMinuteArray(db, meterId, dailyData)
    AddDayArraySummary(db, meterId, overallData)

    return
