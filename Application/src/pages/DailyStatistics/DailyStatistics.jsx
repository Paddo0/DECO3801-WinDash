import React, { useEffect, useState, useContext } from 'react';
import { collection, getDocs } from "firebase/firestore";
import { db } from "../../firebase"; // Ensure this path is correct
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { DailyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';

function DailyStatistics() {
    const { config } = useContext(SettingsContext);
    const [data, setData] = useState([["Time", "Power Consumption"]]);  // Initialize with header

    useEffect(() => {
        const fetchData = async () => {
            const querySnapshot = await getDocs(collection(db, "dailyData")); // Use the correct collection name
            const seriesData = [];
            querySnapshot.forEach(doc => {
                doc.data().seriesData.forEach(entry => {
                    const powerConsumption = entry.Voltage * entry.Intensity; // Calculate power consumption
                    const date = entry.Date.toDate(); // Convert Firestore timestamp to JavaScript Date
                    seriesData.push([date, powerConsumption]);
                });
            });
            seriesData.sort((a, b) => a[0] - b[0]); // Sort by date
            setData([["Time", "Power Consumption"], ...seriesData]);
        };
        fetchData();
    }, []);

    function GetSummaryData() {
        var summaryData = [
            ["Daily Total:", "55.7", "kWh" ], // Update these values based on fetched data
            ["Max Intensity:", "8.0", "kW" ],
            ["Minimum Intensity:", "1.4", "kW" ],
            ["Average Intensity:", "5", "kW" ],
        ];

        return summaryData;
    }

    const usageData = {
        powerUsage: 5.4, // Update this based on fetched data
        usageLimit: config.usageLimits.dailyLimit,
    };

    return (
        <div className="DailyStatistics">
            <UsageGraph title="Daily Statistics" data={data} graphConfig={DailyGraphConfig}/>
            <UsageStatistics title="Daily Summary" summaryData={GetSummaryData()}/>
            {/* Pass 'Daily' as the predictionsInfo to indicate this is for daily predictions */}
            <UsagePredictions predictionsInfo="Daily" usageData={usageData}/>
            <UsageLimit usageData={usageData}/>
        </div>
    );
}
  
export default DailyStatistics;
