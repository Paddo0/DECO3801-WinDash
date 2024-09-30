import constants
from helpers.DatabaseMutationHelper import *
from helpers.DatabaseRetrieveHelper import *
from helpers.DatabaseUpdateHelper import *
from helpers.CsvDataHelper import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def SetupWithHistoricalData(db, meterId, date):
    AddMeter(db, meterId)
    data = ExtractVirtualMeterCsvData(db, meterId, date)

def setupWithNoData(db, meterId):
    ClearAllData(db, meterId)
    AddMeter(db, meterId)

def Resume():
    return

def StartFromPoint():
    return