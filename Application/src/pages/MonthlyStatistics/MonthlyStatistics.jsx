import React, { useEffect, useState, useContext } from 'react';
import { collection, getDocs } from "firebase/firestore";
import { db } from "../../firebase";
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { MonthlyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';

function MonthlyStatistics() {
    const { config } = useContext(SettingsContext);
    const [graphData, setGraphData] = useState([["Time", "Power Consumption"]]);
    const [summaryData, setSummaryData] = useState([]);
    const [usageData, setUsageData] = useState({ powerUsage: 0, usageLimit: config.usageLimits.monthlyLimit });

    useEffect(() => {
        const fetchData = async () => {
            try {
                const querySnapshot = await getDocs(collection(db, "overallData"));
                const dataEntries = [];
                let totalUsage = 0;
                let maxDailyUsage = 0;
                let minDailyUsage = Infinity;
                let avgIntensityTotal = 0;
                let count = 0;

                querySnapshot.forEach((doc) => {
                    const docData = doc.data();
                    docData.overallData.forEach(entry => {
                        const date = entry.Date.toDate();
                        const consumption = entry.TotalConsumption;
                        const avgIntensity = entry.AverageIntensity;

                        dataEntries.push([date, consumption]);

                        totalUsage += consumption;
                        maxDailyUsage = Math.max(maxDailyUsage, consumption);
                        minDailyUsage = Math.min(minDailyUsage, consumption);
                        avgIntensityTotal += avgIntensity;
                        count++;
                    });
                });

                const avgIntensity = avgIntensityTotal / count;

                setGraphData([["Time", "Power Consumption"], ...dataEntries]);

                // Set summaryData based on fetched data
                setSummaryData([
                    ["Total Usage:", totalUsage.toFixed(2), "kWh", config.usageLimits.monthlyLimit],
                    ["Max Daily Usage:", maxDailyUsage.toFixed(2), "kWh", config.usageLimits.monthlyLimit],
                    ["Min Daily Usage:", minDailyUsage.toFixed(2), "kWh", config.usageLimits.monthlyLimit],
                    ["Average Intensity:", avgIntensity.toFixed(2), "kW", "N/A"],
                ]);

                // Set usageData dynamically from Firebase data
                setUsageData({
                    powerUsage: totalUsage,  // The total consumption fetched
                    usageLimit: config.usageLimits.monthlyLimit
                });

            } catch (error) {
                console.error("Error fetching data from Firestore:", error);
            }
        };

        fetchData();
    }, [config.usageLimits.monthlyLimit]);

    return (
        <div className="MonthlyStatistics">
            <UsageGraph title="Monthly Statistics" data={graphData} graphConfig={MonthlyGraphConfig} />
            <UsageStatistics title="Monthly Summary" summaryData={summaryData} />
            <UsagePredictions predictionsInfo={PredictionsInfo.Monthly} usageData={usageData} />
            <UsageLimit usageData={usageData} />
        </div>
    );
}

export default MonthlyStatistics;
