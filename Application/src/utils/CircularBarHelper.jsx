
export const GetColor = (percent) => {
    // Low usage
    if (percent <= 0.8)
    {
        // Green
        return 'rgba(49, 196, 93, 255)'
    }
    // Warning usage
    else if (percent <= 1.0)
    {
        // Yellow / Orange
        return 'rgba(245, 189, 49, 255)'
    }
    // High usage
    else
    {
        // Red
        return 'rgba(252, 40, 40, 255)'
    }
}
