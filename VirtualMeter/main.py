import constants
from helpers.DatabaseMutationHelper import *
from helpers.CsvDataHelper import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def main():
    """
    Main method to call when running project
    """

    # Command input to distinguish between functionality
    command = "set_current_day"

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

    elif command == "add_minute_array":
        AddMinuteArray(db, constants.meterId,
                       list([[
                            constants.minuteTime,
                            constants.minuteIntensity,
                            constants.minuteVoltage
                       ]]))

    elif command == "set_current_day":
        SetCurrentDay(db, constants.meterId,
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


if __name__ == "__main__":
    main()
