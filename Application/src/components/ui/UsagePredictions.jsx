import { useState } from "react";
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";


function UsagePredictions(props) {
    const [prediction, setPrediction] = useState(null);
    const [meterId, setMeterId] = useState(""); // Status for entering meterId

    const handlePredictionClick = async () => {
        console.log("Meter ID being sent:", meterId); 

        const apiEndpoint = props.predictionsInfo === "Daily" 
            ? "http://127.0.0.1:5000/daily-prediction"
            : "http://127.0.0.1:5000/monthly-prediction";

        try {
            const response = await fetch(apiEndpoint, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ meterId })  // send meterId to the backend
            });
            const data = await response.json();
            console.log("Response received:", data);
            if (data.error) {
                console.error("Error from server:", data.error);
            } else {
                setPrediction(data.prediction);
            }
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
                <input 
                    type="text" 
                    value={meterId} 
                    onChange={(e) => setMeterId(e.target.value)} 
                    placeholder="Enter Meter ID" 
                    id="meter-id-input"
                />
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
