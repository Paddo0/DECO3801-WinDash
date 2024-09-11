import { useCallback, useContext, useEffect, useState } from "react";
import { GetDefaultSettings, SettingsContext } from "./SettingsContext";
import MeterDisplay from "./MeterDisplay";
import MeterInput from "../../components/form/MeterInput";

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
      if (config.meterId != null)
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

  //#region Variable Update Handlers

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

  return (
    <div className="Settings">
      <h1>Settings / Configuration</h1>
      <h2>Usage Limit Configuration</h2>

      <div className="SettingsLabel">
        <p>Daily Limit</p>
        <p>Monthly Limit</p>
      </div>

      <div className="SettingsInput">
        <p><input value={config.usageLimits.dailyLimit} onChange={e => updateDailyLimit(e.target.value)}/> kWh</p>
        <p><input value={config.usageLimits.monthlyLimit} onChange={e => updateMonthlyLimit(e.target.value)} /> kWh</p>
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
