
export const PageNames = {
    HOME_PAGE_NAME: "Dashboard",
    DAILY_PAGE_NAME: "Daily_Statistics",
    MONTHLY_PAGE_NAME: "Monthly_Statistics",
    SETTINGS_PAGE_NAME: "Settings",
    INFO_PAGE_NAME: "Help"
};

export const DefaultSettings = {
    meterId: 1003253,
    usageLimits: {
        dailyLimit: 20,
        monthlyLimit: 1000
    }
}

// Date Constants
export const Dates = {
    Today: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate()),
    TodayHours: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), new Date().getHours())
}

// All date data is structured, new Date(Year, Month, Day, Hour, Minute, Second)
export const TimeIntervals = {
    OneHour: {
        minValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), new Date().getHours() - 1),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), new Date().getHours())
    },
    TwelveHour: {
        minValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), new Date().getHours() - 12),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate(), new Date().getHours())
    },
    Daily: {
        minValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate()),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate() + 1)
    },
    FiveDay: {
        minValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate() - 5),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate())
    },
    OneMonth: {
        minValue: new Date(new Date().getFullYear(), new Date().getMonth() - 1, new Date().getDate()),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate())
    },
    ThreeMonth: {
        minValue: new Date(new Date().getFullYear(), new Date().getMonth() - 3),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate())
    },
    SixMonth: {
        minValue: new Date(new Date().getFullYear(), new Date().getMonth() - 6),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate())
    },
    Yearly: {
        minValue: new Date(new Date().getFullYear() - 1, new Date().getMonth(), new Date().getDate()),
        maxValue: new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate())
    }
}

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
    },
    Monthly: {
        InfoText: "Generate the predicted usage for the remainder of the quarter / month using algorithms based on your usage patterns.",
        prediction: 153.4, // Dummy data / default value
    }
}
