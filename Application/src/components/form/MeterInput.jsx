import Popup from 'reactjs-popup';
import MeterImage from "../../assets/images/stock-meter.webp";
import { SettingsContext } from '../../pages/Settings/SettingsContext';
import { useState, useContext, useRef, useEffect } from 'react';
import Keyboard from 'react-simple-keyboard';

/**
 * Defines the meter id input form for users to specify their meter id
 * @returns {React.JSX.Element} Meter Id Input form pop-up
 */
function MeterInput(props)
{
    // Settings config
    const { setConfig } = useContext(SettingsContext);

    // Defining meter input id
    const [ meterId, setMeterId ] = useState("");

    // Updating meter input id on change
    function updateMeterId(newMeterId)
    {
        setMeterId(newMeterId);
    }

    // Updating settings on click
    function updateConfig()
    {
        // Updating meter id only if it is defined
        if (meterId != null && meterId !== "")
        {
            // Saving meterId using config hook and setting to local storage
            setConfig(previousConfig => {
                return {...previousConfig, meterId: meterId}
            })
            localStorage.setItem("meterId", meterId);
        }
    }

    
    // #region Keyboard

    // Defining states
    const [ keyboard, setKeyboard ] = useState(<></>);
    const keyboardRef = useRef(null);
    
    // Adding mouse down listener to close keyboard when clicking away
    useEffect(() => {
      document.addEventListener("mousedown", HandleOutsideClick);
      return () => {
        document.removeEventListener("mousedown", HandleOutsideClick);
      }
    });
  
    // Function to remove keyboard when clicking outside of input / keyboard fields
    const HandleOutsideClick = (e) => {
      if (keyboardRef.current && !keyboardRef.current.contains(e.target))
      {
        RemoveKeyboard();
      }
    }
    
    // Function to make keyboard with bind to given value
    const MakeKeyboard = (e, setValue) => {
      // Clearing old keyboard
      RemoveKeyboard();
  
      // Making new keyboard
      setKeyboard(<Keyboard
        layoutName="default"
        theme="hg-theme-default hg-layout-default keyboard"
        onKeyPress={(button) => OnKeyPressed(button, e, setValue)}
      />);
    };
  
    // Handles all key presses for keyboard
    const OnKeyPressed = (button, e, setValue) => {
      // Handle invalid keys
      if (button === "{shift}" || button === "{lock}" || button === "{space}" || button === "{tab}")
      {
        return;
      }
  
      // Handle backspace
      if (button === "{bksp}")
      {
        setValue(e.target.value.slice(0, -1));
        return;
      }
  
      // Handle enter
      if (button === "{enter}")
      {
        RemoveKeyboard();
        return;
      }
  
      // Handle key pressed
      setValue(e.target.value + button);
    }
  
    // Sets keyboard state to empty component
    const RemoveKeyboard = () => {
      setKeyboard(<></>);
    }
  
    // #endregion

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
                                    
                                    <input value={meterId} onChange={e => updateMeterId(e.target.value)} onFocus={e => MakeKeyboard(e, updateMeterId)} />

                                    <br />

                                    <button className="AddMeterButton" onClick={() => {updateConfig(); close()}}>Submit </button>
                                </div>

                                <div className="modal-add-meter-image">
                                    <img src={MeterImage} alt="Example Meter" />
                                </div>
                            </div>

                            <div ref={keyboardRef} className='keyboardDiv'>
                                {keyboard}
                            </div>
                        </>
                    )
                }
            </Popup>
    );
}

export default MeterInput;
