import { useEffect, useState } from "react";
import { GetColor } from "../../utils/CircularBarHelper";

/**
 * Defines the statistics display component to display statistics to users
 * @returns {React.JSX.Element} Component containing statistics component
 */
function UsageStatistics(props)
{
    // Function to retrieve summary values from props
    function SummaryValues(summary) {
        // Defining initial component
        let values = <></>;
        
        // Iterating over provided summary data
        for (let i = 0; i < summary.display.length; i++)
        {
            // Defining progress bar if limit array defined
            var progressBar = <></>;
            if (summary.limit.length > i)
            {
                // Handling limit being 0
                if (summary.limit[i] > 0)
                {
                    // Creating percentage bar
                    var percentage = summary.display[i][1] / summary.limit[i];
                    progressBar = <progress 
                        value={percentage} 
                        className={GetColor(percentage, true)}
                        style={
                            { 
                                margin: 'auto',
                            }
                        } 
                    />
                }
            }


            // Displaying numbers and labels
            values = <>
                {values}
                <div className="UsageStatisticsDisplay">
                    <div className="UsageStatisticsLabels">
                        <p>{summary.display[i][0]}</p>
                    </div>

                    <div className="UsageStatisticsValues">
                        <p>{summary.display[i][1] + " " + summary.display[i][2]}</p>
                    </div>
                </div>
                
                <div className="UsageStatisticsDisplay">
                    {progressBar}
                </div>
            </>;
        }

        return values;
    }

    // Defining statistics state
    const [ summary, setSummary ] = useState(SummaryValues(props.summaryData));

    // Updating summary on props update
    useEffect(() => {
        setSummary(SummaryValues(props.summaryData));
    }, [props.summaryData]);

    return(
        <div className="UsageStatistics">
            <div className="UsageStatisticsBar">
                <p>{props.title}</p>
            </div>
            
            {summary}
            
        </div>
    )
}

export default UsageStatistics;
