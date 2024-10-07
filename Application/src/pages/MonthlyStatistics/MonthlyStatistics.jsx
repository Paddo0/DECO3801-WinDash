import React, { useState, useContext, useEffect } from 'react';
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { MonthlyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { OverallDataContext } from '../../utils/ContextProvider';
import { CalculateOverallLastNDays } from '../../utils/SummaryHelper';

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

    // Defining usage data state
    const [ usageData, setUsageData ] = useState({
        powerUsage: 0.0, // Update this based on fetched data
        usageLimit: config.usageLimits.monthlyLimit,
    });

    // Handle overall data changes
    useEffect(() => {
        // Updating usage limits
        setUsageData((previousData) => { return {...previousData, powerUsage: CalculateOverallLastNDays(overallData, 30)} });
    }, [overallData]);

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
