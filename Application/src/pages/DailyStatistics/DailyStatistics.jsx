import { useContext } from "react";
import { SettingsContext } from "../Settings/SettingsContext";

/**
 * Base daily statistics page component
 * @returns {React.JSX.Element} Daily statistics component
 */
function DailyStatistics() {
    // Settings Config
    const { config } = useContext(SettingsContext);

    return (
        <div className="DailyStatistics">
            Daily statistics page. {config.usageLimits.monthlyLimit}
        </div>
    );
  }
  
export default DailyStatistics;
