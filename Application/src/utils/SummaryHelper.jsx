
/**
 * Function to return the current usage amount for the day
 * @param {*} dailyData Data to sum usage over
 * @returns kWh used in current day
 */
export const CalculateDailyUsage = (dailyData) => {
    // Handling empty dailyData
    if (dailyData.length <= 1)
    {
        return 0.0;
    }

    // Summing each intensity in daily data
    var total = 0.0;
    for (var i = 1; i < dailyData.length; i++)
    {
        // Adding power usage from each entry in daily data 
        total += dailyData[i][1];
    }

    // Returning total power usage in kWh (divided by 60to convert minutes to hours (Wh) and 1000 to (kWh))
    return total / 60.0;
};

/**
 * Function to return the power usage over the last n days
 * @param {*} overallData Data to sum over for values
 * @param {*} n Number of final entries to sum over
 * @returns Power usage in kWh over the last n days
 */
export const CalculateOverallLastNDays = (overallData, n) => {
    // Checking if overall data is populated with values
    if (overallData.length <= 1 || n <= 0)
    {
        return 0.0;
    }

    // Iterating over the last n days and getting the total usage
    const totalLen = overallData.length;
    var totalUsage = 0.0;
    for (var i = Math.max(totalLen - n, 1); i < totalLen; i++)
    {
        totalUsage += overallData[i][4];
    }

    // Returning the total usage
    return totalUsage;
}
