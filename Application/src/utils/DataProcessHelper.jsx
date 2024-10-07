

/**
 * Function to define the value which is deemed as high usage
 * @param {Array} overallData Database data representing overall daily summaries
 * @param {int} n Number of most recent days to sample from to calculate average max usage
 * @returns 
 */
export const GetHighUsage = (overallData, n) =>
{
    // Checking if overall data is populated with values
    if (overallData.length <= 1 || n <= 0)
    {
        return 0.0;
    }

    // Iterating over the last n days and getting the average highest usage
    const totalLen = overallData.length;
    var highUsage = 0.0;
    for (var i = Math.max(totalLen - n, 0); i < totalLen; i++)
    {
        highUsage += overallData[i][2];
    }

    // Returning average
    return highUsage / n;
}

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
