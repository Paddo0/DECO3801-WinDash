

/**
 * Defines the statistics display component to display statistics to users
 * @returns {React.JSX.Element} Component containing statistics component
 */
function UsageStatistics(props)
{
    return(
        <div className="UsageStatistics">
            <div className="UsageStatisticsBar">
                <p>{props.title}</p>
            </div>
        </div>
    )
}

export default UsageStatistics;
