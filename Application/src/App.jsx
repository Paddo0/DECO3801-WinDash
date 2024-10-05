import Navbar from "./components/ui/Navbar";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Dashboard from "./pages/Dashboard/Dashboard";
import NoPage from "./pages/NoPage/NoPage";
import DailyStatistics from "./pages/DailyStatistics/DailyStatistics";
import MonthlyStatistics from "./pages/MonthlyStatistics/MonthlyStatistics";
import Settings from "./pages/Settings/Settings";
import { PageNames, DailyChartHeaders, OverallChartHeaders } from './data/constants';
import { settings, SettingsContext } from "./pages/Settings/SettingsContext";
import { useState, useEffect } from "react";
import InfoPage from "./pages/InfoPage/InfoPage";
import Slideshow from "./pages/Slideshow/Slideshow";
import { GetDailyData, GetOverallData } from "./utils/DatabaseHelper";
import { DailyDataContext, OverallDataContext } from "./utils/ContextProvider";

/**
 * Default app component to initialize the webpage
 * @returns {React.JSX.Element} Main application component
 */
function App() {
  // Defining settings context
  const [config, setConfig] = useState(settings);

  // Defining database data context
  const [dailyData, setDailyData] = useState(DailyChartHeaders);
  const [overallData, setOverallData] = useState(OverallChartHeaders);

  // Fetching data on load and settings change
  useEffect(() => {
    // Getting data from database
    QueryDatabase();
  }, [config.meterId]);

  // Initializing timer on app definition
  useEffect(() => {
    // Creating interval timer (30s)
    const interval = setInterval(QueryDatabase, 30000);

    // Clean up timer when component deconstructed
    return () => {
      clearInterval(interval);
    }
  });

  // Function to get data from database, call to refresh data
  const QueryDatabase = () => {
    GetDailyData(setDailyData, config.meterId);
    GetOverallData(setOverallData, config.meterId);
  }

  return (
    <SettingsContext.Provider value={{config, setConfig}}>
      <DailyDataContext.Provider value={{dailyData, setDailyData}}>
        <OverallDataContext.Provider value={{overallData, setOverallData}}>
          <div className="App">
            {/* Defining Router */}
            <BrowserRouter>
              <Routes>
                <Route path="/" element={<Navbar />}>
                  <Route index element={<Dashboard />} />
                  <Route path={PageNames.SLIDESHOW_PAGE_NAME} element={<Slideshow />} />
                  <Route path={PageNames.HOME_PAGE_NAME} element={<Dashboard />} />
                  <Route path={PageNames.DAILY_PAGE_NAME} element={<DailyStatistics />} />
                  <Route path={PageNames.MONTHLY_PAGE_NAME} element={<MonthlyStatistics />} />
                  <Route path={PageNames.SETTINGS_PAGE_NAME} element={<Settings />} />
                  <Route path={PageNames.INFO_PAGE_NAME} element={<InfoPage />} />
                  <Route path="*" element={<NoPage />} />
                </Route>
              </Routes>
            </BrowserRouter>

            {/* Main output for routed content is within Navbar.jsx */}
          </div>
        </OverallDataContext.Provider>
      </DailyDataContext.Provider>
    </SettingsContext.Provider>
  );
}

export default App;
