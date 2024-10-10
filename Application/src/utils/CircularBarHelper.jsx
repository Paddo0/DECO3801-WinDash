
export const GetColor = (percent, className = false) => {
    // Low usage
    if (percent <= 0.8)
    {
        // Green
        return className ? 'lowUsage' : 'rgba(49, 196, 93, 255)'
    }
    // Warning usage
    else if (percent <= 1.0)
    {
        // Yellow / Orange
        return className ? 'warningUsage' : 'rgba(245, 189, 49, 255)'
    }
    // High usage
    else
    {
        // Red
        return className ? 'highUsage' : 'rgba(252, 40, 40, 255)'
    }
}
