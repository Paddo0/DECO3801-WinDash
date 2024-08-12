import MeterImage from "../../assets/images/stock-meter.webp";
import RemoveMeterPopup from "./RemoveMeterPopup";
import { useContext } from "react";
import { SettingsContext } from "./SettingsContext";

/**
 * Component to showcase the meter properties of a defined meter
 * @returns {React.JSX.Element} Meter display component
 */
function MeterDisplay()
{
    // Settings config
    const { config } = useContext(SettingsContext);

    return(
        <div className="MeterSettings">
            <div className="MeterLabels">
                <h1>Current Meter Setup</h1>

                <p>&emsp;Model: &emsp; 3AIA4CA010</p>
                <p>&emsp;Id: &emsp; &emsp; {config.meterId}</p>
                <p>&emsp;Year: &emsp; 2024</p>

                <RemoveMeterPopup button={<button className="RemoveButton">Remove</button>} />
            </div>

            <div className="MeterImage">
                <img src={MeterImage} alt="Example Meter" />
            </div>
      </div>
    )
}

export default MeterDisplay;
