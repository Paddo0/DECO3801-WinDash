import constants
from helpers.DatabaseMutationHelper import *
from helpers.DatabaseRetrieveHelper import *
from helpers.DatabaseUpdateHelper import *
from helpers.CsvDataHelper import *
from helpers.VirtualMeterHelper import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def main():
    """
    Main method to call when running project
    """

    # Command input to distinguish between functionality
    command = "get_all_overall"

    # Defines connection to firebase database
    database = Initialize()

    # Runs the command input
    # Uses configurations within constants.py as inputs for commands
    RunCommand(command, database)


def Initialize():
    # Loading credentials
    cred = credentials.Certificate("data/deco-windash-firebase-adminsdk-u0tv1-f05fb8cc6f.json")

    # Initializing app
    app = firebase_admin.initialize_app(cred)

    # Establishing database reference
    database = firestore.client()

    return database


def RunCommand(command, db):
    # Main commands to run Virtual Meter
    if command == "setup_with_historical_data":
        SetupWithHistoricalData(db, constants.meterId, datetime.datetime(2024, 10, 7, 12, 0))

    elif command == "setup_with_no_data":
        setupWithNoData(db, constants.meterId, datetime.datetime(2024, 10, 9, 22, 0))

    elif command == "start_from_point":
        StartFromPoint(db, constants.meterId, datetime.datetime(2024, 10, 7, 23, 57))

    elif command == "resume":
        Resume(db, constants.meterId)

    # Below are commands that are not required for running of thew virtual meter but can still be used for testing/debugging
    # Please change the inputs as required for testing/debugging
    # Please ensure that if you are testing/debugging that you change the value of the meterID input (or change its value in the constants file) so that it is not 1003253 or an existing meterID that you wish not to overwrite/edit.
    elif command == "add_meter":
        AddMeter(db, constants.meterId)

    elif command == "clear_today_data":
        ClearTodayData(db, constants.meterId)

    elif command == "clear_yesterday_data":
        ClearYesterdayData(db, constants.meterId)

    elif command == "add_minute_data":
        AddMinuteData(db, constants.meterId,
                      constants.minuteTime,
                      constants.minuteIntensity,
                      constants.minuteVoltage)

    elif command == "add_minute_array":
        AddMinuteArray(db, constants.meterId,
                       list([[
                            constants.minuteTime,
                            constants.minuteIntensity,
                            constants.minuteVoltage
                       ]]))
        
    elif command == "add_yesterday_minute_array":
        AddYesterdayMinuteArray(db, constants.meterId,
                       list([[
                            constants.minuteTime,
                            constants.minuteIntensity,
                            constants.minuteVoltage
                       ]]))

    elif command == "add_day_summary":
        AddDaySummary(db, constants.meterId,
                      constants.summaryDate,
                      constants.summaryData)

    elif command == "add_day_array_summary":
        AddDayArraySummary(db, constants.meterId,
                           list([[
                               constants.summaryDate,
                               constants.summaryData
                           ]]))

    elif command == "clear_overall_data":
        ClearOverallData(db, constants.meterId)

    elif command == "clear_all_data":
        ClearAllData(db, constants.meterId)

    elif command == "clear_daily_from_time":
        ClearDailyFromTime(db, constants.meterId, constants.minuteTime)

    elif command == "clear_overall_from_date":
        ClearOverallFromDate(db, constants.meterId, datetime.datetime(2024, 9, 14))

    elif command == "get_minute_data":
        print(GetMinuteData(db, constants.meterId, datetime.datetime(2024, 10, 7, 12, 00)))

    elif command == "get_day_summary":
        print(GetDaySummary(db, constants.meterId, datetime.datetime(2024, 10, 6)))

    elif command == "get_latest_minute":
        print(GetLatestMinute(db, constants.meterId))

    elif command == "get_latest_day_summary":
        print(GetLatestDaySummary(db, constants.meterId))

    elif command == "get_all_daily":
        print(GetAllDaily(db, constants.meterId))

    elif command == "get_all_yesterday":
        print(GetAllYesterdayDailyData(db, constants.meterId))

    elif command == "get_all_overall":
        print(GetAllOverall(db, constants.meterId))

if __name__ == "__main__":
    main()
