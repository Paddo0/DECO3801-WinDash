import React, { useCallback, useContext, useEffect, useState } from 'react';
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { PredictionsInfo, TimeIntervals } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { DailyDataContext, OverallDataContext } from '../../utils/ContextProvider';
import { CalculateDailyUsage } from '../../utils/SummaryHelper';
import { CalculateDailySummary } from '../../utils/DataProcessHelper';

/**
 * Base daily statistics page component
 * @returns {React.JSX.Element} Daily Statistics component
 */
function DailyStatistics() {
    // Getting data contexts for page
    const { config } = useContext(SettingsContext);
    const { dailyData } = useContext(DailyDataContext);
    const { overallData } = useContext(OverallDataContext);
    
    const CalculateSummaryData = useCallback( () =>
    {
        // Handling empty data
        if (dailyData.length <= 1)
        {
            return { display: [
                ["Daily Total:", 0, "kWh" ],
                ["Max Intensity:", 0, "kW" ],
                ["Minimum Intensity:", 0, "kW" ],
                ["Average Intensity:", 0, "kW" ],
            ], 
            limit: [1, 1, 1, 1]};
                
        }

        return CalculateDailySummary(dailyData, overallData, 7);
    }, [dailyData, overallData]);
    // Defining summary state
    const [ summaryData, setSummaryData ] = useState(CalculateSummaryData());

    // Defining graph configs
    const graphOptions = 
        [ TimeIntervals.OneHour,
          TimeIntervals.TwelveHour,
          TimeIntervals.Daily];
    const graphOptionsDefault = 2;

    // Defining date state
    const [ date, setDate ] = useState(new Date());
    

    // Defining usage data state
    const [ usageData, setUsageData ] = useState({
        powerUsage: 0.0, // Update this based on fetched data
        usageLimit: config.usageLimits.dailyLimit,
    });

    // Handle daily data changes
    useEffect(() => {
        // Handle date update
        if (dailyData.length > 1)
        {
            setDate(new Date(dailyData.at(-1)[0]));
        }

        // Updating usage limits
        setUsageData((previousData) => { return {...previousData, powerUsage: CalculateDailyUsage(dailyData)} });

        // Updating summary data object
        setSummaryData(CalculateSummaryData());
    }, [dailyData, CalculateSummaryData]);

    return (
        <div className="DailyStatistics">
            <UsageGraph 
                title="Daily Statistics" data={dailyData} 
                graphOptions={graphOptions} 
                graphOptionsDefault={graphOptionsDefault} 
                date={date} 
                defaultSeries={{
                    index: 0, 
                    seriesColor: { 0: { color: '#6167b0' } }
                }} 
            />

            <UsageStatistics 
                title="Daily Summary" 
                summaryData={summaryData}
            />

            <UsagePredictions 
                predictionsInfo={PredictionsInfo.Daily} 
                usageData={usageData}
            />

            <UsageLimit 
                usageData={usageData}
            />
        </div>
    );
}
  
export default DailyStatistics;
