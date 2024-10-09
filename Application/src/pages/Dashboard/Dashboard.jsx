import '../../assets/styles/components/dashboard.css'
import { Link } from 'react-router-dom';
import { n, PageNames } from "../../data/constants";
import RequireMeterId from '../../components/ui/RequireMeterId';
import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import { useContext, useEffect, useState } from 'react';
import { DailyDataContext, OverallDataContext } from '../../utils/ContextProvider';
import { CalculateDailyUsage, CalculateOverallLastNDays } from '../../utils/SummaryHelper';
import { SettingsContext } from '../Settings/SettingsContext';
import { buildStyles, CircularProgressbar } from 'react-circular-progressbar';
import { GetColor } from '../../utils/CircularBarHelper';
import { GetMax } from '../../utils/DataProcessHelper';

/**
 * Base dashboard page component
 * @returns {React.JSX.Element} Dashboard component
 */
function Dashboard() {
  // Defining context variables
  const { dailyData } = useContext(DailyDataContext);
  const { overallData } = useContext(OverallDataContext);
  const { config } = useContext(SettingsContext);

  // Defining states
  const [ dailyUsage, setDailyUsage ] = useState(0);
  const [ overallUsage, setOverallUsage ] = useState(0);

  // Data update effects
  useEffect(() => {
    setDailyUsage(CalculateDailyUsage(dailyData))
    setOverallUsage(CalculateOverallLastNDays(overallData, n));
  }, [dailyData, overallData]);

  // Fun facts slider configuration
  const SliderSettings = {
    dots: true,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 10000,
    nextArrow: <></>,
    prevArrow: <></>
  }

  // Local helper function
  const GetLatest = (data, i) =>
  {
    // Defining latest entry at column i
    var latest = data[data.length - 1][i];

    // Returning 0 if data is empty (latest is the header row)
    if (typeof latest !== 'number')
    {
      return 0.0;
    }
    
    return latest;
  }

  return (
    <div className="dashboard">
      <div className="header">Dashboard</div>
      <div className="main-content">
        <div className="card usage-limits">
          <h3>Usage Limits</h3>
          <div className="limit">
              <CircularProgressbar
                  value={dailyUsage / config.usageLimits.dailyLimit * 100} 
                  text={(dailyUsage / config.usageLimits.dailyLimit * 100).toFixed(1) + '%'}
                  className='progressBar'
                  strokeWidth={10} 
                  styles={buildStyles({
                      pathColor: GetColor((dailyUsage / config.usageLimits.dailyLimit) / 100),
                      textColor: 'rgba(35, 39, 47, 255)'
                  })}
              />
            <p>Daily Limit: {dailyUsage.toFixed(1)} / {config.usageLimits.dailyLimit} kWh</p>
          </div>

          <div className="limit">
              <CircularProgressbar
                  value={overallUsage / config.usageLimits.monthlyLimit * 100} 
                  text={(overallUsage / config.usageLimits.monthlyLimit * 100).toFixed(1) + '%'}
                  className='progressBar'
                  strokeWidth={10} 
                  styles={buildStyles({
                      pathColor: GetColor((overallUsage / config.usageLimits.monthlyLimit)),
                      textColor: 'rgba(35, 39, 47, 255)'
                  })}
              />
            <p>Monthly Limit: {overallUsage.toFixed(1)} / {config.usageLimits.monthlyLimit} kWh</p>

            <RequireMeterId />
          </div>
        </div>

        <div className="card current-usage">
          <h3>Current Usage</h3>
          <div className="limit">
              <CircularProgressbar
                  value={(GetLatest(dailyData, 1) / Math.max(GetLatest(overallData, 1), 0.01, 0.01, GetMax(dailyData, 1))) * 100} 
                  text={(GetLatest(dailyData, 1)).toFixed(1) + ' kW'}
                  className='progressBar'
                  strokeWidth={10} 
                  styles={buildStyles({
                      pathColor: GetColor((GetLatest(dailyData, 1) / Math.max(GetLatest(overallData, 1), 0.01, GetMax(dailyData, 1)))),
                      textColor: 'rgba(35, 39, 47, 255)'
                  })}
              />

            <RequireMeterId />
          </div>
        </div>

        <div className="card setup-meter">
          <h3>Setup Meter</h3>
          
          <Link to={'/' + PageNames.SETTINGS_PAGE_NAME}>
            <button className='configureMeterButton'>Configure Meter</button>
          </Link>
        </div>

        <div className="card alerts">
          <h3>Fun Facts</h3>
          <div className='slider-container'>
            <Slider {...SliderSettings}>

              <div>
                <p>With the power used today you could boil {(dailyUsage / 0.183).toFixed(0)} cups of tea.</p>
              </div>

              <div>
                <p>Todays total usage would be enough energy to drive an electric car {(dailyUsage / 14.1 * 100).toFixed(1)} km.</p>
              </div>

              <div>
                <p>Total usage for today can power an air conditioner for {(dailyUsage / 3.0).toFixed(1)} hours. </p>
              </div>

              <div>
                <p>{(dailyUsage / 0.02).toFixed(0)} phones can be fully charged with todays power usage. </p>
              </div>

              <div>
                <p> Energy used today could power a microwave for {(dailyUsage).toFixed(1)} hours. </p>
              </div>

              <div>
                <p> Consumption of power today can power a fridge for {(dailyUsage / 0.3).toFixed(1)} hours. </p>
              </div>

              <div>
                <p> Todays power can power an LED for {(dailyUsage / 0.01 / 24).toFixed(1)} days. </p>
              </div>

            </Slider>
          </div>

          <RequireMeterId />
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
