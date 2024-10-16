import { useContext } from "react";
import { buildStyles, CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { SettingsContext } from "../../pages/Settings/SettingsContext";
import { GetColor } from "../../utils/CircularBarHelper";

function UsagePredictions(props) {
    // Defining settings context
    const { config } = useContext(SettingsContext);

    // Defining function to map to predictions results
    const ConvertTokW = (x) => {
        // Assuming 240V & converting from W to kW
        return x * 0.24;
    }

    // Async function to access API on click
    const handlePredictionClick = async () => {
        // Specifying which prediction to make
        const apiEndpoint = props.predictionsInfo.predictionType === "daily" ? 
                            "http://127.0.0.1:5000/daily-prediction" : "http://127.0.0.1:5000/monthly-prediction";

        try {
            // Calling API
            const response = await fetch(apiEndpoint, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ meterId: config.meterId })
            });

            // Defining data as json object
            const data = await response.json();
            console.log(data);
            
            // Checking for errors
            if (data.error) {
                console.error("Error from server:", data.error);
            } else {
                // Saving values and mapping intensity to kW
                if (props.predictionsInfo.predictionType === "daily")
                {
                    props.predictionsSet(data.prediction.map(ConvertTokW));
                }
                else
                {
                    props.predictionsSet(data.prediction);
                }
            }
        } catch (error) {
            console.error("Error fetching prediction:", error);
        }
    };

    // Function to get sum of prediction data
    const CalculatePredictedUsage = (prediction, numeric = false) =>
    {
        // Handling default values
        if (prediction == null)
        {
            // Returning numeric default value based on indicator value
            if (numeric)
            {
                return 0.0;
            }
            else
            {
                return "--";
            }
        }

        // Calculating daily predictions percentage
        if (props.predictionsInfo.predictionType === "daily")
        {
            // Summing predictions
            var sum = 0.0;
            for (var i = 0; i < props.remainingHours; i++)
            {
                sum += prediction[i];
            }
            return sum.toFixed(2);
        }
        else
        {
            return prediction.toFixed(2);
        }
        
    }

    const CalculateProgressPercentage = (prediction, actualUsage, limitUsage) => {
        // Calculating predicted usage
        const predictedUsage = CalculatePredictedUsage(prediction, true)

        // Returning 0 if no predictions have been made
        if (predictedUsage === 0.0)
        {
            return 0.0;
        }

        // Calculated percentage
        return ((Number(predictedUsage) + actualUsage) / limitUsage * 100).toFixed(1);
    }

    return (
        <div className="UsagePredictions">
            <div className="UsagePredictionsColumn">
                <div className="UsagePredictionsBar">
                    <p>Usage Predictions</p>
                </div>
                <p>{props.predictionsInfo?.InfoText || "No InfoText available"}</p> 
                <div className="UsagePredictionsButton">
                    <button onClick={handlePredictionClick}>Generate Prediction</button>
                </div>
            </div>

            <div className="UsagePredictionsColumn">
                <h1>Predicted Usage:</h1>
                <h3>{CalculatePredictedUsage(props.prediction) + " kWh"}</h3>
            </div>

            <div className="UsagePredictionsColumn">
                <h1>Predicted Usage Limit:</h1>
                <div className="UsagePredictionsDisplay">
                    <div className='UsagePredictionsContent'>
                        <div className='UsagePredictionsText'>
                            <h3>{(CalculateProgressPercentage(props.prediction, props.usageData.powerUsage, props.usageData.usageLimit) * props.usageData.usageLimit / 100).toFixed(2) + " / " + (props.usageData?.usageLimit || "--") + " kWh"}</h3>
                        </div>
                    </div>
                    <div className='UsagePredictionsProgressBar'>
                        <CircularProgressbar
                            value={CalculateProgressPercentage(props.prediction, props.usageData.powerUsage, props.usageData.usageLimit)}
                            text={`${CalculateProgressPercentage(props.prediction, props.usageData.powerUsage, props.usageData.usageLimit)} %`}
                            styles={buildStyles({
                                pathColor: GetColor(CalculateProgressPercentage(props.prediction, props.usageData.powerUsage, props.usageData.usageLimit) / 100),
                                textColor: 'rgba(35, 39, 47, 255)'
                            })}
                            strokeWidth={10}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default UsagePredictions;
