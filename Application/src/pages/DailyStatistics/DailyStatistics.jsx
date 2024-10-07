import React, { useContext, useEffect, useState } from 'react';
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { PredictionsInfo, TimeIntervals } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { DailyDataContext } from '../../utils/ContextProvider';
import { CalculateDailyUsage } from '../../utils/SummaryHelper';

/**
 * Base daily statistics page component
 * @returns {React.JSX.Element} Daily Statistics component
 */
function DailyStatistics() {
    // Getting data contexts for page
    const { config } = useContext(SettingsContext);
    const { dailyData } = useContext(DailyDataContext);
    
    function GetSummaryData() {
        var summaryData = [
            ["Daily Total:", "55.7", "kWh" ], // Update these values based on fetched data
            ["Max Intensity:", "8.0", "kW" ],
            ["Minimum Intensity:", "1.4", "kW" ],
            ["Average Intensity:", "5", "kW" ],
        ];

        return summaryData;
    }

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
    }, [dailyData]);

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
                summaryData={GetSummaryData()}
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
