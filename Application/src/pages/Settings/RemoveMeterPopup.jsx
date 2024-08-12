import Popup from 'reactjs-popup';
import { useContext } from "react";
import { SettingsContext } from "./SettingsContext";

/**
 * Component to let the user confirm that they want to delete their meter id
 * @returns {React.JSX.Element} Meter removal popup
 */
function RemoveMeterPopup(props)
{
    // Settings config
    const { setConfig } = useContext(SettingsContext);
    
    // State change handler removing meter if from config and local storage
    function RemoveMeterId()
    {
        // Clearing meterId using config hook and clearing from local storage
        setConfig(previousConfig => {
            return {...previousConfig, meterId: null}
        })
        localStorage.removeItem("meterId");
    }

    return(
        <Popup trigger={props.button} modal nested>
                {
                    close => (
                        <>
                            <div className="BackgroundDimmer" />

                            <div className="modal-confirmation">
                                <button className="close" onClick={close}>
                                    &times;
                                </button>

                                <div className="modal-confirmation-content">
                                    <p>Are you sure you want to delete your saved Meter Id?</p>
                                    <br />
                                    (Your energy history won't be deleted just the current access to the meter.)

                                    <br />
                                
                                    <button className="RemoveButton" onClick={RemoveMeterId}>
                                        Remove
                                    </button>

                                    <button onClick=
                                        {() => close()}>
                                            No
                                    </button>
                                </div>
                            </div>
                        </>
                    )
                }
            </Popup>
    )
}
export default RemoveMeterPopup;
