import datetime


def UpdateDailyDataDifference(db, meterId: str, timeDifference: datetime):
    """
    Shifts the date of every date data in the dailyData table (including currentDate)
    :param db: Database reference to update data for
    :param meterId: Document identifier representing meter to update data for
    :param timeDifference: The difference in time to shift each of the entries by
    """
    # Get the current daily data for the meter
    doc_ref = db.collection("dailyData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        daily_data = doc.to_dict()

        # Update the currentDate by shifting with the time difference
        if 'currentDate' in daily_data:
            current_date = daily_data['currentDate']
            daily_data['currentDate'] = current_date + timeDifference

        # Update each entry in the seriesData array by shifting the date with the time difference
        if 'seriesData' in daily_data:
            updated_series_data = []
            for entry in daily_data['seriesData']:
                entry['Date'] = entry['Date'] + timeDifference
                updated_series_data.append(entry)

            # Save the updated data back to Firestore
            doc_ref.update({
                'currentDate': daily_data['currentDate'],
                'seriesData': updated_series_data
            })

    return


def UpdateMonthlyDataDifference(db, meterId: str, dateDifference: datetime):
    """
    Shifts the date of every date data within the overallData table
    :param db: Database reference to update data for
    :param meterId: Document identifier representing meter to update data for
    :param dateDifference: The difference in time to shift each of the entries by
    """
    # Get the current overall data for the meter
    doc_ref = db.collection("overallData").document(meterId)
    doc = doc_ref.get()

    if doc.exists:
        overall_data = doc.to_dict()

        # Update each entry in the data array by shifting the date with the date difference
        if 'data' in overall_data:
            updated_data = []
            for entry in overall_data['data']:
                entry['Date'] = entry['Date'] + dateDifference
                updated_data.append(entry)

            # Save the updated data back to Firestore
            doc_ref.update({'data': updated_data})

    return


def UpdateAllDatetimeData(db, meterId: str, currentDate: datetime):
    """
    Sets the date data for all tables around a specified point, the given time represents the start of the day to
    pivot the data about; i.e. the overallData's latest entry will be the day before the date given, and the dailyData
    table will start the current seriesData from the given time
    :param db: Database reference to update data for
    :param meterId: Document identifier representing meter to update data for
    :param currentDate: The start of the day to set a meter's date data to, ensure this date is set to midnight at the
                        start of the day to act as a starting point for the dailyData seriesData
    """
    # This is the function to actually calculate the time difference, make sure the difference calculated for the
    # overallData has the most recent entry as the day before the current date and for the currentDate variable to be
    # based on midnight for the user. We may need to have a discussion based on the start of days within the database
    # since firebase stores it as UTC+10 which is AEST which should be fine but is something that should be kept in
    # mind

    # Update dailyData
    doc_ref_daily = db.collection("dailyData").document(meterId)
    doc_daily = doc_ref_daily.get()

    if doc_daily.exists:
        daily_data = doc_daily.to_dict()

        # Calculate the time difference for daily data
        if 'seriesData' in daily_data and daily_data['seriesData']:
            last_entry = daily_data['seriesData'][-1]  # Get the most recent minute data entry
            last_date = last_entry['Date']
            time_difference = currentDate - last_date

            # Update dailyData with the time difference
            UpdateDailyDataDifference(db, meterId, time_difference)

    # Update overallData
    doc_ref_overall = db.collection("overallData").document(meterId)
    doc_overall = doc_ref_overall.get()

    if doc_overall.exists:
        overall_data = doc_overall.to_dict()

        # Calculate the date difference for overall data
        if 'data' in overall_data and overall_data['data']:
            last_summary = overall_data['data'][-1]  # Get the most recent day summary
            last_date = last_summary['Date']
            date_difference = currentDate - last_date

            # Update overallData with the date difference
            UpdateMonthlyDataDifference(db, meterId, date_difference)

    return
