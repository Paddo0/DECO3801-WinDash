import Navbar from "./components/ui/Navbar";
import { BrowserRouter, Routes, Route} from "react-router-dom";
import Dashboard from "./pages/Dashboard/Dashboard";
import NoPage from "./pages/NoPage/NoPage";
import DailyStatistics from "./pages/DailyStatistics/DailyStatistics";
import { PageNames } from './data/constants';

/**
 * Default app component to initialize the webpage
 * @returns {React.JSX.Element} Main application component
 */
function App() {
  return (
    <div className="App">
      {/* Defining Router */}
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navbar />}>
            <Route index element={<Dashboard />} />
            <Route path={PageNames.HOME_PAGE_NAME} element={<Dashboard />} />
            <Route path={PageNames.DAILY_PAGE_NAME} element={<DailyStatistics />} />
            <Route path="*" element={<NoPage />} />
          </Route>
        </Routes>
      </BrowserRouter>

      {/* Main output for routed content is within Navbar.jsx */}
    </div>
  );
}

export default App;
