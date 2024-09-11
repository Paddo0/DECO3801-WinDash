import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { DailyGraphConfig, PredictionsInfo } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { useContext } from 'react';

/**
 * Base daily statistics page component
 * @returns {React.JSX.Element} Daily statistics component
 */
function DailyStatistics() {
    // Settings config
    const { config } = useContext(SettingsContext);

    // Dummy Data
    const data = [["Time", "Power Consumption"],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 0, 0, 0), 1.8],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 1, 45, 0), 1.2],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 3, 45, 0), 1.5],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 5, 30, 0), 3.0],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 7, 15, 0), 2.5],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 9, 30, 0), 6.0],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 10, 30, 0), 7.0],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 14, 15, 0), 8.0],
          [new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), 18, 0, 0), 6.5]];

    // Summary Dummy Data
    function GetSummaryData() {
        var summaryData = [
            ["Daily Total:", "55.7", "kWh" ],
            ["Max Intensity:", "8.0", "kW" ],
            ["Minimum Intensity:", "1.4", "kW" ],
            ["Average Intensity:", "5", "kW" ],
        ];

        return summaryData;
    }

    // Limit dummy data
    const usageData = {
        powerUsage: 5.4,
        usageLimit: config.usageLimits.dailyLimit,
    };

    return (
        <div className="DailyStatistics">
            <UsageGraph title="Daily Statistics" data={data} graphConfig={DailyGraphConfig}/>

            <UsageStatistics title="Daily Summary" summaryData={GetSummaryData()}/>

            <UsagePredictions predictionsInfo={PredictionsInfo.Daily} usageData={usageData} />
            
            <UsageLimit usageData={usageData} />
        </div>
    );
}
  
export default DailyStatistics;
