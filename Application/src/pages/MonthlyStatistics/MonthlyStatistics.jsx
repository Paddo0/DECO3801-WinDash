import React, { useState, useContext } from 'react';
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { MonthlyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { OverallDataContext } from '../../utils/ContextProvider';

function MonthlyStatistics() {
    const { config } = useContext(SettingsContext);
    const { overallData } = useContext(OverallDataContext);
    
    // State for graph data and summary data
    const [ summaryData ] = useState([
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

    const usageData = {
        powerUsage: 803.4, // Example value; update if needed from Firestore
        usageLimit: config.usageLimits.monthlyLimit,
    };

    return (
        <div className="MonthlyStatistics">
            <UsageGraph title="Monthly Statistics" data={overallData} graphConfig={MonthlyGraphConfig} />
            <UsageStatistics title="Monthly Summary" summaryData={summaryData} />
            <UsagePredictions predictionsInfo={PredictionsInfo.Monthly} usageData={usageData} />
            <UsageLimit usageData={usageData} />
        </div>
    );
}

export default MonthlyStatistics;
