
// Number of days calculate back for monthly statistics
export const n = 30;

export const PageNames = {
    SLIDESHOW_PAGE_NAME: "Slideshow",
    HOME_PAGE_NAME: "Dashboard",
    DAILY_PAGE_NAME: "Daily_Statistics",
    MONTHLY_PAGE_NAME: "Monthly_Statistics",
    SETTINGS_PAGE_NAME: "Settings",
    INFO_PAGE_NAME: "Help"
};

// Default settings when values haven't been specified
export const DefaultSettings = {
    meterId: "",
    usageLimits: {
        dailyLimit: 20,
        monthlyLimit: 1000
    }
}

// Default headings for charts
export const DailyChartHeaders = [["Time", "Power Consumption"]];
export const OverallChartHeaders = [["Date", "Average Intensity", "Maximum Intensity", "Minimum Intensity", "Total Consumption"]];

// Date Constants
export var Dates = {
    Today: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate()),
    TodayHours: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), new Date().getHours()),
    OffsetTime: new Date(0, 0, 0, 10)
}

// All date data is structured, new Date(Year, Month, Day, Hour, Minute, Second)
export var TimeIntervals = {
    OneHour: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear(), date.getMonth(), date.getDate(), date.getHours() - 1, date.getMinutes()),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate(), date.getHours(), date.getMinutes()),
                format: "hh:mm a"
                }
            },
        name: "1 Hour",
        index: 0
    },
    TwelveHour: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear(), date.getMonth(), date.getDate(), date.getHours() - 12, date.getMinutes()),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate(), date.getHours(), date.getMinutes()),
                format: "hh:mm a"
                }
            },
        name: "12 Hour",
        index: 1
    },
    Daily: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear(), date.getMonth(), date.getDate()),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1),
                format: "hh:mm a"
                }
            },
        name: "Daily",
        index: 2
    },
    FiveDay: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() - 5 + 1),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1),
                format: "dd MMM"
                }
            },
        name: "5 Day",
        index: 0
    },
    OneMonth: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear(), date.getMonth() - 1, date.getDate() + 1),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1),
                format: "dd MMM"
                }
            },
        name: "1 Month",
        index: 1
    },
    ThreeMonth: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear(), date.getMonth() - 3),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1),
                format: "dd MMM"
                }
            },
        name: "3 Months",
        index: 2
    },
    SixMonth: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear(), date.getMonth() - 6),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1),
                format: "dd MMM"
                }
            },
        name: "6 Months",
        index: 3
    },
    Yearly: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear() - 1, date.getMonth(), date.getDate()),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1),
                format: "dd MMM"
                }
            },
        name: "1 Year",
        index: 4
    },
    ThreeYears: { 
        function: (date) => 
            { return{
                minValue: new Date(date.getFullYear() - 3, date.getMonth(), date.getDate()),
                maxValue: new Date(date.getFullYear(), date.getMonth(), date.getDate() + 1),
                format: "MMM YYYY"
                }
            },
        name: "3 Years",
        index: 5
    },
}

// Series constants
export const DataSeries = [
    {
        name: "All Series",
        seriesName: "",
        index: 0,
        seriesColor: {
            0: { color: '#f1ca3a' },
            1: { color: '#e7711b' },
            2: { color: '#6f9654' },
            3: { color: '#6167b0' }
        }
    },
    {
        name: "Average",
        seriesName: "Average Intensity",
        index: 1,
        seriesColor: { 0: { color: '#f1ca3a' } }
    },
    {
        name: "Max Usage",
        seriesName: "Maximum Intensity",
        index: 2,
        seriesColor: { 0: { color: '#e7711b' } }
    },
    {
        name: "Min Usage",
        seriesName: "Minimum Intensity",
        index: 3,
        seriesColor: { 0: { color: '#6f9654' } }
    },
    {
        name: "Total Usage",
        seriesName: "Total Consumption",
        index: 4,
        seriesColor: { 0: { color: '#6167b0' } }
    }
]

export const DailyGraphConfig = {
    minValue: TimeIntervals.Daily.minValue,
    maxValue: TimeIntervals.Daily.maxValue,
    format: "hh:mm a"
}

export const MonthlyGraphConfig = {
    minValue: TimeIntervals.SixMonth.minValue,
    maxValue: TimeIntervals.SixMonth.maxValue,
    format: "dd MMM"
}

export const PredictionsInfo = {
    Daily:
    {
        InfoText: "Generate the predicted usage for the remainder of the day using our AI Model trained from your usage patterns.",
        prediction: 6.2, // Dummy data / default value
        predictionType: "daily"
    },
    Monthly: {
        InfoText: "Generate the predicted usage for the remainder of the quarter / month using algorithms based on your usage patterns.",
        prediction: 153.4, // Dummy data / default value
        predictionType: "monthly"
    }
}
