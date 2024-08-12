import { useContext } from "react";
import { SettingsContext } from "../Settings/SettingsContext";

/**
 * Base dashboard page component
 * @returns {React.JSX.Element} Dashboard component
 */
function Dashboard() {
  // Settings Config
  const { config } = useContext(SettingsContext);

  return (
    <div className="Dashboard">
       Dashboard page. Limit: {config.usageLimits.dailyLimit}
    </div>
  );
}

export default Dashboard;
