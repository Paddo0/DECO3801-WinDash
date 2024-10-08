import { db } from "../firebase";
import { collection, getDocs, query, where, documentId } from "firebase/firestore";
import { DailyChartHeaders, OverallChartHeaders } from "../data/constants";

// Asynchronous function to get daily data
export const GetDailyData = async (setData, meterId) => {
    // Returning if meterId isn't valid
    if (meterId == null || meterId === "")
    {
        setData(DailyChartHeaders);
        return;
    }

    // Specifying collection to get from database
    var dailyDataCollection = collection(db, "dailyData");
    dailyDataCollection = query(dailyDataCollection, where(documentId(), '==', meterId))

    // Getting data from firebase, and waiting for response
    const dailyDataSnapshot = await getDocs(dailyDataCollection);

    // Mapping firebase data object to js list
    var dailyDataList = dailyDataSnapshot.docs.map(doc => doc.data());

    // Saving data if there are results that match meterId
    if (dailyDataList.length > 0)
    {
        // Map values to [dates, watts]
        dailyDataList = dailyDataList[0]["seriesData"];
        dailyDataList = dailyDataList.map(entry => [entry["Date"].toDate(), entry["Intensity"] * entry["Voltage"] / 1000.0]);

        // Saving array data with header
        setData([DailyChartHeaders[0], ...dailyDataList]);
    }
};

// Asynchronous function to get monthly data
export const GetOverallData = async (setData, meterId) => {
    // Returning if meterId isn't valid
    if (meterId == null || meterId === "")
    {
        setData(OverallChartHeaders);
        return;
    }

    // Specifying collection to get from database
    var overallDataCollection = collection(db, "overallData");
    overallDataCollection = query(overallDataCollection, where(documentId(), '==', meterId))

    // Getting data from firebase, and waiting for response
    const overallDataSnapshot = await getDocs(overallDataCollection);

    // Mapping firebase data object to js list
    var overallDataList = overallDataSnapshot.docs.map(doc => doc.data());

    if (overallDataList.length > 0)
    {
        // Mapping values to [date, ave, max, min, total]
        overallDataList = overallDataList[0]["data"];
        overallDataList = overallDataList.map(entry => [
            entry["Date"].toDate(),
            entry["AverageIntensity"],
            entry["MaximumIntensity"],
            entry["MinimumIntensity"],
            entry["TotalConsumption"]
        ]);
        
        // Saving array data with header
        setData([OverallChartHeaders[0], ...overallDataList]);
    }
};

