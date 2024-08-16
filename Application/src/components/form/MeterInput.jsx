import Popup from 'reactjs-popup';
import MeterImage from "../../assets/images/stock-meter.webp";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { useState, useContext } from 'react';

/**
 * Defines the meter id input form for users to specify their meter id
 * @returns {React.JSX.Element} Meter Id Input form pop-up
 */
function MeterInput(props)
{
    // Settings config
    const { setConfig } = useContext(SettingsContext);

    // Defining meter input id
    const [ meterId, setMeterId ] = useState();

    // Updating meter input id on change
    function updateMeterId(newMeterId)
    {
        setMeterId(newMeterId);
    }

    // Updating settings on click
    function updateConfig()
    {
        // Updating meter id only if it is defined
        if (meterId != null)
        {
            // Saving meterId using config hook and setting to local storage
            setConfig(previousConfig => {
                return {...previousConfig, meterId: meterId}
            })
            localStorage.setItem("meterId", meterId);
    }
    }

    return (
        <Popup trigger={props.button} modal nested>
                {
                    close => (
                        <>
                            <div className="BackgroundDimmer" />

                            <div className="modal-add-meter">
                                <button className="close" onClick={close}>
                                    &times;
                                </button>

                                <div className="modal-add-meter-content">
                                    <h2>Add Meter Id:</h2>

                                    <p>Add the meter id visible on the Win-Dash meter to link functionality to the application.</p>

                                    <br />
                                    
                                    <input onChange={e => updateMeterId(e.target.value)} />

                                    <br />

                                    <button className="AddMeterButton" onClick={updateConfig}>Submit </button>
                                </div>

                                <div className="modal-add-meter-image">
                                    <img src={MeterImage} alt="Example Meter" />
                                </div>
                            </div>
                        </>
                    )
                }
            </Popup>
    );
}

export default MeterInput;
