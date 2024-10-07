import { useCallback, useContext, useEffect, useState, useRef} from "react";
import { GetDefaultSettings, SettingsContext } from "./SettingsContext";
import MeterDisplay from "./MeterDisplay";
import MeterInput from "../../components/form/MeterInput";
import Keyboard from 'react-simple-keyboard';
import 'react-simple-keyboard/build/css/index.css';

/**
 * Base settings page component
 * @returns {React.JSX.Element} Settings component
 */
function Settings() {
  // Settings config
  const { config, setConfig } = useContext(SettingsContext);

  // Meter display callback
  const DefineMeterDisplay = useCallback(() => 
  {
      // Default meter id input
      var display = 
        <div>
          Missing meter id information. Please provide meter id to link to WinDash system.

          <br />

          <MeterInput button={<button className="AddMeterButton" >Add Meter Id</button>} />
        </div>;
  
      // Defining meter display if meter id is defined
      if (config.meterId != null && config.meterId !== "")
      {
        display = <MeterDisplay />;
      }
  
      return display;
  }, [config.meterId]);

  // Meter settings display state
  const [meterDisplay, setMeterDisplay] = useState(DefineMeterDisplay());

  // Resets config back to constants values
  function ResetToDefaults()
  {
    // Setting local storage values
    localStorage.clear();

    // Setting context values
    setConfig(() => {return GetDefaultSettings()});
  }

  // #region Variable Update Handlers

  // Handling daily limit change
  const updateDailyLimit = (newLimit) => {

    // Setting local storage
    localStorage.setItem("dailyLimit", newLimit);

    // Setting context
    setConfig(previousConfig => {
      return { ...previousConfig, usageLimits: {dailyLimit: newLimit, monthlyLimit: previousConfig.usageLimits.monthlyLimit} }
    });
  }

  // Handling daily limit change
  const updateMonthlyLimit = (newLimit) => {

    // Setting local storage
    localStorage.setItem("monthlyLimit", newLimit);

    // Setting context
    setConfig(previousConfig => {
      return { ...previousConfig, usageLimits: {dailyLimit: previousConfig.usageLimits.dailyLimit, monthlyLimit: newLimit} }
    });
  }

  // Effect handling the changing of meter id
  useEffect(() => {
    setMeterDisplay(() => DefineMeterDisplay());
  }, [config, DefineMeterDisplay]);

  //#endregion

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
      theme="keyboard hg-theme-default hg-layout-default"
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
    <div className="Settings">
      <h1>Settings / Configuration</h1>
      <h2>Usage Limit Configuration</h2>

      <div className="SettingsLabel">
        <p>Daily Limit</p>
        <p>Monthly Limit</p>
      </div>

      <div className="SettingsInput">
        <p><input value={config.usageLimits.dailyLimit} onChange={e => updateDailyLimit(e.target.value)} onFocus={e => MakeKeyboard(e, updateDailyLimit)} /> kWh</p>
        <p><input value={config.usageLimits.monthlyLimit} onChange={e => updateMonthlyLimit(e.target.value)} onFocus={e => MakeKeyboard(e, updateMonthlyLimit)}  /> kWh</p>
      </div>

      <div ref={keyboardRef} className="keyboardDiv">
        {keyboard}
      </div>

      <hr />

      <h2>Meter Configuration</h2>
      
      {meterDisplay}

      <hr />
      <h2>Data Configuration</h2>
      <button className="ResetButton" onClick={ResetToDefaults}>Reset to defaults</button>
    </div>
  );
}

export default Settings;
