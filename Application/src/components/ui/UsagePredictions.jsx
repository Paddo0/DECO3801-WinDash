import { useState } from "react";
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

function UsagePredictions(props) {
    const [prediction, setPrediction] = useState(null);

    const handlePredictionClick = async () => {
        try {
            const response = await fetch("http://127.0.0.1:5000/pred");
            const data = await response.json();
            console.log("Response received:", data);
            setPrediction(data.prediction);
        } catch (error) {
            console.error("Error fetching prediction:", error);
        }
    };

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
                <h3>{prediction !== null ? `${prediction} kWh` : "No prediction generated yet"}</h3>
            </div>

            <div className="UsagePredictionsColumn">
                <h1>Predicted Usage Limit:</h1>
                <div className="UsagePredictionsDisplay">
                    <div className='UsagePredictionsContent'>
                        <div className='UsagePredictionsText'>
                            <h3>{(prediction !== null ? (prediction + props.usageData?.powerUsage || 0).toFixed(1) : "--") + " / " + (props.usageData?.usageLimit || "--") + " kWh"}</h3>
                        </div>
                    </div>
                    <div className='UsagePredictionsProgressBar'>
                        <CircularProgressbar
                            value={prediction !== null ? ((prediction + (props.usageData?.powerUsage || 0)) / (props.usageData?.usageLimit || 1)) * 100 : 0}
                            text={prediction !== null ? `${((prediction + (props.usageData?.powerUsage || 0)) / (props.usageData?.usageLimit || 1) * 100).toFixed(1)}%` : "--"}
                            strokeWidth={10}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default UsagePredictions;
