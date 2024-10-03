import React, { useEffect, useState, useContext } from 'react';
import { collection, getDocs } from "firebase/firestore";
import { db } from "../../firebase";
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { DailyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';

function DailyStatistics() {
    const { config } = useContext(SettingsContext);
    const [data, setData] = useState([["Time", "Power Consumption"]]);
    const [summaryData, setSummaryData] = useState([]);
    const [usageData, setUsageData] = useState({ powerUsage: 0, usageLimit: config.usageLimits.dailyLimit });

    useEffect(() => {
        const fetchData = async () => {
            const querySnapshot = await getDocs(collection(db, "dailyData"));
            const seriesData = [];
            let totalPower = 0;
            let maxIntensity = 0;
            let minIntensity = Infinity;
            let intensitySum = 0;
            let count = 0;

            querySnapshot.forEach(doc => {
                doc.data().seriesData.forEach(entry => {
                    const powerConsumption = entry.Voltage * entry.Intensity;
                    const intensity = entry.Intensity;
                    const date = entry.Date.toDate();

                    // Collect series data for graph
                    seriesData.push([date, powerConsumption]);

                    // Calculate values for summary
                    totalPower += powerConsumption;
                    maxIntensity = Math.max(maxIntensity, intensity);
                    minIntensity = Math.min(minIntensity, intensity);
                    intensitySum += intensity;
                    count++;
                });
            });

            seriesData.sort((a, b) => a[0] - b[0]);
            setData([["Time", "Power Consumption"], ...seriesData]);

            // Calculate average intensity
            const avgIntensity = intensitySum / count;

            // Set summaryData based on fetched data
            setSummaryData([
                ["Daily Total:", totalPower.toFixed(2), "kWh", config.usageLimits.dailyLimit],
                ["Max Intensity:", maxIntensity.toFixed(2), "kW", "N/A"],
                ["Minimum Intensity:", minIntensity.toFixed(2), "kW", "N/A"],
                ["Average Intensity:", avgIntensity.toFixed(2), "kW", "N/A"]
            ]);

            // Update usage data
            setUsageData({ powerUsage: totalPower, usageLimit: config.usageLimits.dailyLimit });
        };

        fetchData();
    }, [config.usageLimits.dailyLimit]);

    return (
        <div className="DailyStatistics">
            <UsageGraph title="Daily Statistics" data={data} graphConfig={DailyGraphConfig}/>
            <UsageStatistics title="Daily Summary" summaryData={summaryData}/>
            <UsagePredictions predictionsInfo={PredictionsInfo.Daily} usageData={usageData}/>
            <UsageLimit usageData={usageData}/>
        </div>
    );
}

export default DailyStatistics;
