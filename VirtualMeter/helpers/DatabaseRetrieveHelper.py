import datetime


def GetMinuteData(db, meterId: str, time: datetime) -> (float, float):
    return


def GetDaySummary(db, meterId: str, date: datetime) -> (float, float, float, float):
    return


def GetLatestMinute(db, meterId: str) -> (datetime, float, float):
    return


def GetLatestDaySummary(db, meterId: str) -> (datetime, float, float, float, float):
    return


def GetAllDaily(db, meterId: str) -> (list[datetime, float, float]):
    return


def GetAllOverall(db, meterId: str) -> (list[datetime, float, float, float, float]):
    return
