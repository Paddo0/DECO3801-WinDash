import React from "react";
import { DailyChartHeaders, OverallChartHeaders } from "../data/constants";

// Creating daily data context
export const DailyDataContext = React.createContext(
    // Default headers
    DailyChartHeaders
);

// Creating overall data context
export const OverallDataContext = React.createContext(
    // Default headers
    OverallChartHeaders
);
