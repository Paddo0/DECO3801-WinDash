import { useCallback, useContext, useEffect, useState } from "react";
import { SettingsContext } from "../../pages/Settings/SettingsContext";
import MeterInput from "../form/MeterInput";

/**
 * Component to add to div to block content until the meterId has been input
 * @returns {React.JSX.Element} A div component to overlay content on invalid meter ID's
 */
function RequireMeterId()
{
    // Defining context
    const { config } = useContext(SettingsContext);
    
    // Function defining overlay component
    const CalculateOverlayComponent = useCallback(() => {
        // Return empty component when meterId is valid
        if (config.meterId != null && config.meterId !== "")
        {
            return (<></>);
        }
        
        // Calculate overlay if meterId isn't valid
        return(
            <div className="meterIdOverlay">
                <div className="meterIdForm">
                    <p>Meter Id required to display data.</p>

                    <MeterInput button={<button className="AddMeterButton" >Add Meter Id</button>} />
                </div>
            </div>
        );
    }, [config.meterId]);

    // Defining overlay state
    const [ overlayComponent, setOverlayComponent] = useState(CalculateOverlayComponent());

    // Effect to remove overlay when meter configured
    useEffect(() => {
        setOverlayComponent(CalculateOverlayComponent());
    }, [config.meterId, CalculateOverlayComponent]);

    return(
        <>
            {overlayComponent}
        </>
    );
}

export default RequireMeterId;
