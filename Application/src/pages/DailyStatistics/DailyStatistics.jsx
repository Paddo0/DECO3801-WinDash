import React, { useContext } from 'react';
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { DailyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { DailyDataContext } from '../../utils/ContextProvider';

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

    const usageData = {
        powerUsage: 5.4, // Update this based on fetched data
        usageLimit: config.usageLimits.dailyLimit,
    };

    return (
        <div className="DailyStatistics">
            <UsageGraph title="Daily Statistics" data={dailyData} graphConfig={DailyGraphConfig}/>
            <UsageStatistics title="Daily Summary" summaryData={GetSummaryData()}/>
            <UsagePredictions predictionsInfo={PredictionsInfo.Daily} usageData={usageData}/>
            <UsageLimit usageData={usageData}/>
        </div>
    );
}
  
export default DailyStatistics;
