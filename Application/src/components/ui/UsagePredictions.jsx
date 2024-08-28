

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

                <p>{props.InfoText}</p>

                <div className="UsagePredictionsButton">
                    <button>Generate Prediction</button>
                </div>
            </div>

            <div className="UsagePredictionsColumn">
                
            </div>
            
            <div className="UsagePredictionsColumn">
                
            </div>
        </div>
    )
}

export default UsagePredictions;
