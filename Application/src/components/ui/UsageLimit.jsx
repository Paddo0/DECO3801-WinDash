import React from 'react';
import { CircularProgressbar } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { Link } from 'react-router-dom';
import { PageNames } from '../../data/constants';

/**
 * Defines the limit display component to display usage limit data
 * @returns {React.JSX.Element} Component containing usage limit component
 */
function UsageLimit({ usageData }) {
    const { powerUsage, usageLimit } = usageData;
    const percentage = usageLimit ? ((powerUsage / usageLimit) * 100).toFixed(1) : 0;

    return (
        <div className="UsageLimit">
            <div className="UsageLimitBar">
                <p>Usage Limits</p>
            </div>
                
            <div className="UsageLimitDisplay">
                <div className='UsageLimitContent'>
                    <div className='UsageLimitText'>
                        {powerUsage} / {usageLimit ? usageLimit : "N/A"} kWh
                    </div>

                    <div className='UsageLimitButton'>
                        <Link to={'/' + PageNames.SETTINGS_PAGE_NAME}>
                            <button>Set Limit</button>
                        </Link>
                    </div>
                </div>
                
                <div className='UsageLimitProgressBar'>
                    <CircularProgressbar 
                        value={percentage} 
                        text={`${percentage}%`} 
                        strokeWidth={10} 
                    />
                </div>
            </div>
        </div>
    );
}

export default UsageLimit;
