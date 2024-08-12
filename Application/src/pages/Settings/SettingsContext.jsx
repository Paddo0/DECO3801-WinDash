import React from "react";
import { DefaultSettings } from "../../data/constants";

// Default settings object
export const settings = GetDefaultSettings();

/**
 * Handles the retreival of data from storage, if undefined, return default value
 * @param {string} variableName 
 * @param {*} defaultValue 
 * @returns 
 */
function getLocalValue(variableName, defaultValue)
{
    // Getting value from local storage
    var value = localStorage.getItem(variableName);

    // Setting default value when variable is null
    if (value == null)
    {
        value = defaultValue;

        // Saving default value to local storage
        localStorage.setItem(variableName, value);
    }

    return value;
}

// Returns the default settings from constants if not already defined in local storage
// If not defined in local storage, set values to defaults
export function GetDefaultSettings()
{
    const defaultSettings = {
        meterId: getLocalValue("meterId", DefaultSettings.meterId),
        usageLimits: {
            dailyLimit: getLocalValue("dailyLimit", DefaultSettings.usageLimits.dailyLimit),
            monthlyLimit: getLocalValue("monthlyLimit", DefaultSettings.usageLimits.monthlyLimit)
        }
    }

    return defaultSettings;
};

// Creating settings context
export const SettingsContext = React.createContext(
    // Default Settings
    settings
);
