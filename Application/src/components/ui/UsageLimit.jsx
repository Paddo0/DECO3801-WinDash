import {buildStyles, CircularProgressbar} from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { Link } from 'react-router-dom';
import { PageNames } from '../../data/constants';
import { GetColor } from '../../utils/CircularBarHelper';

/**
 * Defines the limit display component to display usage limit data
 * @returns {React.JSX.Element} Component containing usage limit component
 */
function UsageLimit(props) {
    const percentUsage = props.usageData.powerUsage / props.usageData.usageLimit * 100;

    return(
        <div className="UsageLimit">
            <div className="UsageLimitBar">
                <p>Usage Limits</p>
            </div>
                
            <div className="UsageLimitDisplay">
                <div className='UsageLimitContent'>
                    <div className='UsageLimitText'>
                        {props.usageData.powerUsage.toFixed(1)} / {props.usageData.usageLimit} kWh
                    </div>

                    <div className='UsageLimitButton'>
                        <Link to={'/' + PageNames.SETTINGS_PAGE_NAME}>
                            <button>Set Limit</button>
                        </Link>
                    </div>
                </div>
                
                <div className='UsageLimitProgressBar'>
                    <CircularProgressbar 
                        value={percentUsage} 
                        text={percentUsage.toFixed(1) + '%'}
                        strokeWidth={10} 
                        styles={buildStyles({
                            pathColor: GetColor(percentUsage / 100),
                            textColor: 'rgba(35, 39, 47, 255)'
                        })}
                    />
                </div>
            </div>
        </div>
    )
}

export default UsageLimit;
