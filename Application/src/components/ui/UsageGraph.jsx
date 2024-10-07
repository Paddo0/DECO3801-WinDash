import Dropdown from "./Dropdown";
import { Chart } from "react-google-charts";
import RequireMeterId from "./RequireMeterId";


/**
 * Defines the graph display component to display usage data
 * @returns {React.JSX.Element} Component containing graph component
 */
function UsageGraph(props)
{
    return(
        <div className="UsageGraph">
            <div className="UsageGraphBar">
                <div className='UsageGraphControl'>
                    <p>{props.title}</p>
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown placeholder="Select Option" />
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown placeholder="Select Option" />
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown placeholder="Select Option" />
                </div>
            </div>

            <div className="UsageGraphDisplay">
                <Chart 
                    chartType="AreaChart"
                    data={props.data}
                    width="100%"
                    height="100%"
                    options={
                        {
                            backgroundColor: "#f5f5f5",
                            hAxis: {    
                                format: props.graphConfig.format,
                                minValue: props.graphConfig.minValue,
                                maxValue: props.graphConfig.maxValue,
                            },
                            
                            vAxis: { 
                                title: "Power Consumption",
                                minValue: 0,
                                titleTextStyle: {
                                    fontSize: '1.4vw',
                                    fontName: 'Helvetica',
                                },
                            },

                            chartArea: {
                                width: "75%",
                                height: "80%",
                                right: "15%",
                            }
                        }
                    }
                />
            </div>
            
            <RequireMeterId />

        </div>
    )
}

export default UsageGraph;