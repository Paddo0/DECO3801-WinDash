

/**
 * Defines the statistics display component to display statistics to users
 * @returns {React.JSX.Element} Component containing statistics component
 */
function UsageStatistics(props)
{
    // Function to retrieve summary data from props
    function SummaryLabels(summary){
        // Defining initial component
        let labels = <></>;
        
        // Iterating over provided summary data
        for (let i = 0; i < summary.length; i++){
            labels = <>
                {labels} 
                {<p>{summary[i][0]}</p>}
            </>;
        }

        return labels;
    }

    // Function to retrieve summary values from props
    // TODO - These functions will need to be changed when summary becomes dynamic and requires hooks
    function SummaryValues(summary) {
        // Defining initial component
        let values = <></>;
        
        // Iterating over provided summary data
        for (let i = 0; i < summary.length; i++){
            values = <>
                {values} 
                {<p>{summary[i][1] + " " + summary[i][2]}</p>}
            </>;
        }

        return values;
    }

    return(
        <div className="UsageStatistics">
            <div className="UsageStatisticsBar">
                <p>{props.title}</p>
            </div>
            
            <div className="UsageStatisticsDisplay">
                <div className="UsageStatisticsLabels">
                    {SummaryLabels(props.summaryData)}
                </div>

                <div className="UsageStatisticsValues">
                    {SummaryValues(props.summaryData)}
                </div>
            </div>
        </div>
    )
}

export default UsageStatistics;
