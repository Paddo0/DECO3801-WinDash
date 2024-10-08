import React, { useState, useContext, useEffect } from 'react';
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { DataSeries, PredictionsInfo, TimeIntervals } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { OverallDataContext } from '../../utils/ContextProvider';
import { CalculateOverallLastNDays } from '../../utils/SummaryHelper';
import { CalculateOverallSummary } from '../../utils/DataProcessHelper';

function MonthlyStatistics() {
    const { config } = useContext(SettingsContext);
    const { overallData } = useContext(OverallDataContext);
    
    // Function to calculate summary data object
    function CalculateSummaryData() 
    {
        // Handling empty data
        if (overallData.length <= 1)
        {
            // Returning default summary object
            return { display: [
                ["Past Month Averages:", " ", " "], // Invisible character to pad space
                ["Total Consumption:", 0, "kWh" ],
                ["Maximum Intensity:", 0, "kW" ],
                ["Minimum Intensity:", 0, "kW" ],
                ["Average Intensity:", 0, "kW" ],
            ], 
            limit: [0, 1, 1, 1, 1]};
        }

        return CalculateOverallSummary(overallData, 30);
    }

    // Defining summary state
    const [ summaryData, setSummaryData ] = useState(CalculateSummaryData());

    // Defining graph configs
    const graphOptions = 
        [ 
            TimeIntervals.FiveDay,
            TimeIntervals.OneMonth,
            TimeIntervals.ThreeMonth,
            TimeIntervals.SixMonth,
            TimeIntervals.Yearly,
            TimeIntervals.ThreeYears
        ];
    const graphOptionsDefault = 3;
    const dataOptions = DataSeries

    // Defining date state
    const [ date, setDate ] = useState(new Date());
    

    // Defining usage data state
    const [ usageData, setUsageData ] = useState({
        powerUsage: 0.0, // Update this based on fetched data
        usageLimit: config.usageLimits.monthlyLimit,
    });

    // Handle overall data changes
    useEffect(() => {
        // Handle date update
        if (overallData.length > 1)
        {
            setDate(new Date(overallData.at(-1)[0]));
        }
        // Updating usage limits
        setUsageData((previousData) => { return {...previousData, powerUsage: CalculateOverallLastNDays(overallData, 30)} });

        // Updating summary data
        setSummaryData(CalculateSummaryData());
    }, [overallData]);

    return (
        <div className="MonthlyStatistics">
            <UsageGraph 
                title="Monthly Statistics" 
                data={overallData} 
                graphOptions={graphOptions} 
                graphOptionsDefault={graphOptionsDefault} 
                dataOptions={dataOptions} 
                date={date}
                defaultSeries={DataSeries[0]}
            />

            <UsageStatistics 
                title="Monthly Summary" 
                summaryData={summaryData} 
            />

            <UsagePredictions 
                predictionsInfo={PredictionsInfo.Monthly} 
                usageData={usageData} 
            />

            <UsageLimit 
                usageData={usageData} 
            />
        </div>
    );
}

export default MonthlyStatistics;
