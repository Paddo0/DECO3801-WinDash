import Dropdown from "./Dropdown";
import { Chart } from "react-google-charts";
import RequireMeterId from "./RequireMeterId";
import { useCallback, useEffect, useState } from "react";
import { GetSeriesData } from "../../utils/DataProcessHelper";


/**
 * Defines the graph display component to display usage data
 * @returns {React.JSX.Element} Component containing graph component
 */
function UsageGraph(props)
{
    // Defining graph states
    const [ graphConfig, setGraphConfig ] = useState(props.graphOptions[props.graphOptionsDefault].function(props.date));
    const [ graphConfigIndex, setGraphConfigIndex ] = useState(props.graphOptionsDefault);
    const [ seriesState, setSeriesState ] = useState(props.defaultSeries);
    const [ dataState, setDataState ] = useState(props.data);
    const [ colors, setColors ] = useState(props.defaultSeries.seriesColor);

    // Data state helper
    const setSeriesDataState = useCallback((series) => {
        // Setting data with series slice
        setSeriesState(series);

        // Getting series data based on state selected
        setDataState(GetSeriesData(props.data, series.index));

        setColors(series.seriesColor);
    }, [props.data]);

    // Effects to update data
    useEffect(() => {
        setSeriesDataState(seriesState);
    }, [props.data, setSeriesDataState, seriesState]);

    // Effect to update date
    useEffect(() => {
        setGraphConfig(props.graphOptions[graphConfigIndex].function(props.date));
    }, [props.date, props.graphOptions, graphConfigIndex]);

    // Set graph config function that specifies date to function option returned
    const setGraphConfigState = useCallback((config) => {
        setGraphConfig(config.function(props.date));
        setGraphConfigIndex(config.index);
    }, [props.date]);


    return(
        <div className="UsageGraph">
            <div className="UsageGraphBar">
                <div className='UsageGraphControl'>
                    <p>{props.title}</p>
                </div>

                <div className='UsageGraphControl'>
                    {/* Empty control to maintain spacings */}
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown setValue={setSeriesDataState} options={props.dataOptions} defaultValue={props.dataOptionsDefault} />
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown setValue={setGraphConfigState} options={props.graphOptions} defaultValue={props.graphOptionsDefault} />
                </div>
            </div>

            <div className="UsageGraphDisplay">
                <Chart 
                    chartType="LineChart"
                    data={dataState}
                    width="100%"
                    height="100%"
                    options={
                        {
                            lineWidth: 2,
                            series: colors,
                            backgroundColor: "#f5f5f5",
                            hAxis: {    
                                format: graphConfig.format,
                                viewWindow: {
                                    min: graphConfig.minValue,
                                    max: graphConfig.maxValue,
                                }
                            },
                            
                            vAxis: { 
                                title: "Power Consumption",
                                viewWindow: {
                                    min: 0,
                                },
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