import '../../assets/styles/components/dashboard.css'
import { Link } from 'react-router-dom';
import { PageNames } from "../../data/constants";

/**
 * Base dashboard page component
 * @returns {React.JSX.Element} Dashboard component
 */
function Dashboard() {

  return (
    <div className="dashboard">
      <div className="header">Dashboard</div>
      <div className="main-content">
        <div className="card usage-limits">
          <h3>Usage Limits</h3>
          <div className="limit">
            <div className="circle">70%</div>
            <p>Daily Limit: 25.7 / 35.0 kWh</p>
          </div>
          <div className="limit">
            <div className="circle">25%</div>
            <p>Monthly Limit: 550.8 / 2000.0 kWh</p>
          </div>
        </div>

        <div className="card current-usage">
          <h3>Current Usage</h3>
          <div className="circle">5.8 Amps</div>
        </div>

        <div className="card setup-meter">
          <h3>Setup Meter</h3>
          <button>Configure Meter</button>
        </div>

        <div className="card alerts">
          <h3>Alerts</h3>
          <p>!!! Daily usage limit almost reached !!!</p>
          <p>!!! High current energy usage !!!</p>
        </div>
      </div>

      <div className="footer">
        <Link to={'/' + PageNames.DAILY_PAGE_NAME}>
          <button className="footer-button">Daily Statistics</button>
        </Link>
        <Link to={'/' + PageNames.MONTHLY_PAGE_NAME}>
          <button className="footer-button">Monthly Statistics</button>
        </Link>
        <Link to={'/' + PageNames.SETTINGS_PAGE_NAME}>
          <button className="footer-button">Settings</button>
        </Link>
        <Link to={'/' + PageNames.INFO_PAGE_NAME}>
          <button className="footer-button">Info / Help</button>
        </Link>
      </div>
    </div>
  );
}

export default Dashboard;
