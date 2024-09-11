import { CircularProgressbar } from "react-circular-progressbar";

/**
 * Defines the prediction display component to display predicted usage
 * @returns {React.JSX.Element} Component containing prediction component
 */
function UsagePredictions(props)
{

    return(
        <div className="UsagePredictions">
            <div className="UsagePredictionsColumn">
                <div className="UsagePredictionsBar">
                    <p>Usage Predictions</p>
                </div>

                <p>{props.predictionsInfo.InfoText}</p>

                <div className="UsagePredictionsButton">
                    <button>Generate Prediction</button>
                </div>
            </div>

            <div className="UsagePredictionsColumn">
                <h1>Predicted Usage:</h1>

                <h3>{props.predictionsInfo.prediction} kWh</h3>
            </div>
            
            <div className="UsagePredictionsColumn">
                <h1>Predicted Usage Limit:</h1>
                
                <div className="UsagePredictionsDisplay">
                    <div className='UsagePredictionsContent'>
                        <div className='UsagePredictionsText'>
                            <h3>{(props.predictionsInfo.prediction + props.usageData.powerUsage).toFixed(1)} / {props.usageData.usageLimit} kWh</h3>
                        </div>
                    </div>
                    
                    <div className='UsagePredictionsProgressBar'>
                        <CircularProgressbar 
                            value={(props.predictionsInfo.prediction + props.usageData.powerUsage) / props.usageData.usageLimit * 100} 
                            text={((props.predictionsInfo.prediction + props.usageData.powerUsage) / props.usageData.usageLimit * 100).toFixed(1) + '%'} 
                            strokeWidth={10} 
                        />
                    </div>
                </div>
            </div>
        </div>
    )
}

export default UsagePredictions;
