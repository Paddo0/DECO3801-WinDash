################################################################################
# A file to organize all constants for the virtual meter in a shared location. #
################################################################################
import datetime

# File path for consumption data file
dataFilepath = "data/household_power_consumption.csv"

# MeterId to index database tables
#meterId = "1003253"
meterId = "test19"

# Input Settings #
# Daily Data
# Defines individual minute time value for daily data
minuteTime = datetime.datetime(2024, 9, 12, 5, 25)

# Defines the minute average intensity for daily data
minuteIntensity = 16.8

# Defines the minute average voltage for daily data
minuteVoltage = 244.8

# Defines the current date for daily data
currentDate = datetime.datetime(2024, 9, 12)

# Overall Data
# Defines the date to set the day data summary for
summaryDate = datetime.datetime(2006, 12, 15)

# Defines the data summary for overallData
summaryData = (15.4, 27.5, 8.2, 11.2)
