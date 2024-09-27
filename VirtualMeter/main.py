import constants
from helpers.DatabaseMutationHelper import *
from helpers.DatabaseRetrieveHelper import *
from helpers.DatabaseUpdateHelper import *
from helpers.CsvDataHelper import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def main():
    """
    Main method to call when running project
    """

    # Command input to distinguish between functionality
    command = "clear_daily_from_time"

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
    if command == "add_meter":
        AddMeter(db, constants.meterId)

    elif command == "add_minute_data":
        AddMinuteData(db, constants.meterId,
                      constants.minuteTime,
                      constants.minuteIntensity,
                      constants.minuteVoltage)

    elif command == "clear_today_data":
        ClearTodayData(db, constants.meterId)

    elif command == "clear_yesterday_data":
        ClearYesterdayData(db, constants.meterId)

    elif command == "add_minute_array":
        """
        AddMinuteArray(db, constants.meterId,
                       list([[
                            constants.minuteTime,
                            constants.minuteIntensity,
                            constants.minuteVoltage
                       ]]))
        """
        array = [(datetime.datetime(2024, 9, 12, 5, 20), 10, 250),
                 (datetime.datetime(2024, 9, 12, 5, 21), 20, 250),
                 (datetime.datetime(2024, 9, 12, 5, 22), 20, 250),
                 (datetime.datetime(2024, 9, 12, 5, 23), 20, 250),
                 (datetime.datetime(2024, 9, 12, 5, 24), 30, 250),
                 ]

        # (time, intensity, voltage)

        AddMinuteArray(db, constants.meterId, array)
        
    elif command == "add_yesterday_minute_array":
        AddYesterdayMinuteArray(db, constants.meterId,
                       list([[
                            constants.minuteTime,
                            constants.minuteIntensity,
                            constants.minuteVoltage
                       ]]))

    elif command == "set_current_day":
        SetCurrentDay(db, constants.meterId,
                      constants.currentDate)
        
    elif command == "set_yesterday_day":
        SetYesterdayDay(db, constants.meterId,
                      constants.currentDate)

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

    elif command == "load_from_csv":
        LoadCsvData(db, constants.meterId, constants.dataFilepath)

    elif command == "clear_daily_from_time":
        #ClearDailyFromTime(db, constants.meterId, constants.minuteTime)
        ClearDailyFromTime(db, constants.meterId, datetime.datetime(2024, 9, 12, 5, 22))

    elif command == "clear_overall_from_date":
        ClearOverallFromDate(db, constants.meterId, constants.summaryDate)

    elif command == "get_minute_data":
        print(GetMinuteData(db, constants.meterId, constants.minuteTime))

    elif command == "get_day_summary":
        print(GetDaySummary(db, constants.meterId, constants.summaryDate))

    elif command == "get_latest_minute":
        print(GetLatestMinute(db, constants.meterId))

    elif command == "get_latest_day_summary":
        print(GetLatestDaySummary(db, constants.meterId))

    elif command == "get_all_daily":
        print(GetAllDaily(db, constants.meterId))

    elif command == "get_all_overall":
        print(GetAllOverall(db, constants.meterId))

    elif command == "update_daily_data_difference":
        UpdateDailyDataDifference(db, constants.meterId, constants.currentDate - constants.minuteTime)

    elif command == "update_daily_data_difference":
        UpdateMonthlyDataDifference(db, constants.meterId, constants.currentDate - constants.summaryDate)

    elif command == "update_all_datetime_data":
        UpdateAllDatetimeData(db, constants.meterId, constants.currentDate)


if __name__ == "__main__":
    main()
