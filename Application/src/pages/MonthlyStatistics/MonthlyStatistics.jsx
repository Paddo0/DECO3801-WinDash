import { useContext } from "react";
import { SettingsContext } from "../Settings/SettingsContext";
import UsageGraph from "../../components/ui/UsageGraph";
import UsageStatistics from "../../components/ui/UsageStatistics";
import UsagePredictions from "../../components/ui/UsagePredictions";
import UsageLimit from "../../components/ui/UsageLimit";

/**
 * Base daily statistics page component
 * @returns {React.JSX.Element} Daily statistics component
 */
function MonthlyStatistics() {
    // Settings Config
    const { config } = useContext(SettingsContext);

    return (
        <div className="MonthlyStatistics">
            <UsageGraph title="Monthly Statistics" />

            <UsageStatistics title="Monthly Summary" />

            <UsagePredictions />
            
            <UsageLimit />
        </div>
    );
}
  
export default MonthlyStatistics;
