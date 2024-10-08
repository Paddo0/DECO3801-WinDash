

/**
 * Function to define the value which is deemed as high usage
 * @param {Array} overallData Database data representing overall daily summaries
 * @param {int} n Number of most recent days to sample from to calculate average max usage
 * @returns 
 */
export const GetAverageUsage = (overallData, n, i) =>
{
    // Checking if overall data is populated with values
    if (overallData.length <= 1 || n <= 0)
    {
        return 0.0;
    }

    // Iterating over the last n days and getting the average highest usage
    const totalLen = overallData.length;
    var highUsage = 0.0;
    for (var j = Math.max(totalLen - n, 1); j < totalLen; j++)
    {
        highUsage += overallData[j][i];
    }

    // Returning average
    return highUsage / n;
}

/**
 * Gets slice of overall data for specified column index
 * @param {*} overallData Data to get slice from
 * @param {*} index Column to retrieve
 * @returns 
 */
export const GetSeriesData = (overallData, index) =>
{
    // Checking if overall data is populated with values
    if (overallData.length <= 1 || index === 0)
    {
        return overallData;
    }

    // Singling out the index slice specified
    var seriesData = overallData.map(entry => [entry[0], entry[index]]);
    return seriesData;
}

/**
 * Daily summary data function
 * @param {*} dailyData Daily data to summarize
 * @param {*} overallData Overall data to get previous days' averages
 * @param {*} n Amount of previous days to average over
 * @returns Daily summary data object
 */
export const CalculateDailySummary = (dailyData, overallData, n) => {

    // Calculating summary statistics
    var total = 0;
    var max = 0;
    var min = Infinity;
    for (var i = 1; i < dailyData.length; i++)
    {
        total += dailyData[i][1];
        max = Math.max(max, dailyData[i][1]);
        min = Math.min(min, dailyData[i][1]);

    }

    // Converting to minute contributions
    var average = total / dailyData.length;
    total /= 60.0;
    
    // Creating summary object
    return { display: [
        ["Daily Total:", total.toFixed(1), "kWh" ],
        ["Max Intensity:", max.toFixed(1), "kW" ],
        ["Minimum Intensity:", min.toFixed(1), "kW" ],
        ["Average Intensity:", average.toFixed(1), "kW" ],
    ], 
    limit: [
        // Calculating average of variables over last n days
        GetAverageUsage(overallData, n, 4), 
        GetAverageUsage(overallData, n, 2), 
        GetAverageUsage(overallData, n, 3), 
        GetAverageUsage(overallData, n, 1)]};
}

/**
 * Overall summary data function
 * @param {*} overallData Overall data to summarize
 * @param {*} n Amount of previous days to average over
 * @returns Overall summary data object
 */
export const CalculateOverallSummary = (overallData, n) => {

    // Calculating summary statistics
    var total = 0;
    var max = 0;
    var min = 0;
    var average = 0;
    for (var i = Math.max(overallData.length - n, 1); i < overallData.length; i++)
    {
        total += overallData[i][4];
        max += overallData[i][2];
        min += overallData[i][3];
        average += overallData[i][1];
    }

    // Converting to averages
    total /= n;
    max /= n;
    min /= n;
    average /= n;
    
    // Creating summary object
    return { display: [
        ["Past Month Averages:", " ", " "], // Invisible character to pad space
        ["Total Consumption:", total.toFixed(1), "kWh" ],
        ["Maximum Intensity:", max.toFixed(1), "kW" ],
        ["Minimum Intensity:", min.toFixed(1), "kW" ],
        ["Average Intensity:", average.toFixed(1), "kW" ],
    ], 
    limit: [
        // Calculating average of variables over last n days
        // multiplying by arbitrary 1.5 otherwise half the time values will be over
        0,
        GetAverageUsage(overallData, overallData.length, 4) * 1.5, 
        GetAverageUsage(overallData, overallData.length, 2) * 1.5, 
        GetAverageUsage(overallData, overallData.length, 3) * 1.5, 
        GetAverageUsage(overallData, overallData.length, 1) * 1.5]};
}
