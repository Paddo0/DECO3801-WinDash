import React, { useEffect, useState, useContext } from 'react';
import { collection, getDocs } from "firebase/firestore"; // Import getDocs for fetching data
import { db } from "../../firebase"; // Ensure correct Firebase config path
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { MonthlyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';

function MonthlyStatistics() {
    const { config } = useContext(SettingsContext);
    
    // State for graph data and summary data
    const [graphData, setGraphData] = useState([["Time", "Power Consumption"]]);
    const [summaryData, setSummaryData] = useState([
        ["This Quarter:", "", "-"],
        ["Total Usage:", "0", "kWh"],
        ["Max Daily Usage:", "0", "kWh"],
        ["Min Daily Usage:", "0", "kWh"],
        ["Average Intensity:", "0", "kW"],
        ["Overall:", "", "-"],
        ["Usage:", "0", "kWh"],
        ["Max Daily Usage:", "0", "kWh"],
        ["Min Daily Usage:", "0", "kWh"],
        ["Average Intensity:", "0", "kW"],
    ]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Fetch the overallData collection from Firestore
                const querySnapshot = await getDocs(collection(db, "overallData"));
                const dataEntries = [];
                let totalUsage = 0;
                let maxDailyUsage = 0;
                let minDailyUsage = Infinity;
                let avgIntensityTotal = 0;
                let count = 0;
    
                querySnapshot.forEach((doc) => {
                    const docData = doc.data();
                    // Check if `data` array exists in the document
                    if (docData && Array.isArray(docData.data)) {
                        docData.data.forEach(entry => {
                            const date = entry.Date.toDate();
                            const consumption = entry.TotalConsumption;
                            const avgIntensity = entry.AverageIntensity;
                            
                            // Push data for the graph
                            dataEntries.push([date, consumption]);
                            
                            // Calculate for summary
                            totalUsage += consumption;
                            maxDailyUsage = Math.max(maxDailyUsage, consumption);
                            minDailyUsage = Math.min(minDailyUsage, consumption);
                            avgIntensityTotal += avgIntensity;
                            count++;
                        });
                    } else {
                        console.warn(`Document ${doc.id} does not contain valid overallData`);
                    }
                });
    
                const avgIntensity = avgIntensityTotal / count;
    
                // Set the graph data
                setGraphData([["Time", "Power Consumption"], ...dataEntries]);
    
                // Set the summary data
                setSummaryData([
                    ["This Quarter:", "", "-"],
                    ["Total Usage:", `${totalUsage.toFixed(2)} kWh`],
                    ["Max Daily Usage:", `${maxDailyUsage.toFixed(2)} kWh`],
                    ["Min Daily Usage:", `${minDailyUsage.toFixed(2)} kWh`],
                    ["Average Intensity:", `${avgIntensity.toFixed(2)} kW`],
                    ["Overall:", "", "-"],
                    ["Usage:", `${totalUsage.toFixed(2)} kWh`],
                    ["Max Daily Usage:", `${maxDailyUsage.toFixed(2)} kWh`],
                    ["Min Daily Usage:", `${minDailyUsage.toFixed(2)} kWh`],
                    ["Average Intensity:", `${avgIntensity.toFixed(2)} kW`],
                ]);
            } catch (error) {
                console.error("Error fetching data from Firestore:", error);
            }
        };
    
        fetchData();
    }, []); // Empty dependency array ensures this runs once when the component mounts
    
    const usageData = {
        powerUsage: 803.4, // Example value; update if needed from Firestore
        usageLimit: config.usageLimits.monthlyLimit,
    };

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
