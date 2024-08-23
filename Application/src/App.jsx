import Navbar from "./components/ui/Navbar";
import { BrowserRouter, Routes, Route} from "react-router-dom";
import Dashboard from "./pages/Dashboard/Dashboard";
import NoPage from "./pages/NoPage/NoPage";
import DailyStatistics from "./pages/DailyStatistics/DailyStatistics";
import MonthlyStatistics from "./pages/MonthlyStatistics/MonthlyStatistics";
import Settings from "./pages/Settings/Settings";
import { PageNames } from './data/constants';
import { settings, SettingsContext } from "./pages/Settings/SettingsContext";
import { useState } from "react";

/**
 * Default app component to initialize the webpage
 * @returns {React.JSX.Element} Main application component
 */
function App() {
  const [config, setConfig] = useState(settings);

  return (
    <SettingsContext.Provider value={{config, setConfig}}>
      <div className="App">
        {/* Defining Router */}
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Navbar />}>
              <Route index element={<Dashboard />} />
              <Route path={PageNames.HOME_PAGE_NAME} element={<Dashboard />} />
              <Route path={PageNames.DAILY_PAGE_NAME} element={<DailyStatistics />} />
              <Route path={PageNames.MONTHLY_PAGE_NAME} element={<MonthlyStatistics />} />
              <Route path={PageNames.SETTINGS_PAGE_NAME} element={<Settings />} />
              <Route path="*" element={<NoPage />} />
            </Route>
          </Routes>
        </BrowserRouter>

        {/* Main output for routed content is within Navbar.jsx */}
      </div>
    </SettingsContext.Provider>
  );
}

export default App;
