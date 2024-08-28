import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";
import { MonthlyGraphConfig, PredictionsInfoText } from "../../data/constants";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { useContext } from 'react';

/**
 * Base daily statistics page component
 * @returns {React.JSX.Element} Daily statistics component
 */
function MonthlyStatistics() {
    // Settings config
    const { config } = useContext(SettingsContext);

    // Dummy Data
    const data = [["Time", "Power Consumption"],
                [new Date(), 1]];
    
    // Summary Data
    function GetSummaryData() {
        var summaryData = [
            ["This Quarter:", "", "-" ],
            ["Total Usage:", "2007.7", "kWh" ],
            ["Max Daily Usage:", "16.8", "kWh" ],
            ["Min Daily Usage:", "8.4", "kWh" ],
            ["Average Intensity:", "5.5", "kW" ],
            ["Overall:", "", "-" ],
            ["Usage:", "15628.7", "kWh" ],
            ["Max Daily Usage:", "21.4", "kWh" ],
            ["Min Daily Usage:", "6.8", "kWh" ],
            ["Average Intensity:", "7.2", "kW" ],
        ];

        return summaryData;
    }

    // Limit dummy data
    const usageData = {
        powerUsage: 803.4,
        usageLimit: config.usageLimits.monthlyLimit,
    };

    return (
        <div className="MonthlyStatistics">
            <UsageGraph title="Monthly Statistics" data={data} graphConfig={MonthlyGraphConfig} />

            <UsageStatistics title="Monthly Summary" summaryData={GetSummaryData()} />

            <UsagePredictions InfoText={PredictionsInfoText.MonthlyInfoText} />
            
            <UsageLimit usageData={usageData} />
        </div>
    );
}
  
export default MonthlyStatistics;
