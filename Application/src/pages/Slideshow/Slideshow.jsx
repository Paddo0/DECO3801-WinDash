import { useContext, useEffect, useState } from "react";
import Images from "../../utils/SlideshowImport";
import { PageNames } from "../../data/constants";
import { Link } from "react-router-dom";
import { DailyDataContext, OverallDataContext } from "../../utils/ContextProvider";
import { GetAverageUsage } from "../../utils/DataProcessHelper";

/**
 * Base slideshow page component
 * @returns {React.JSX.Element} Slideshow component
 */
function Slideshow() {
    // Getting data context for page
    const { dailyData } = useContext(DailyDataContext);
    const { overallData } = useContext(OverallDataContext);

    // Defining states
    const [ highUsage, setHighUsage ] = useState(0.0);
    const [ displayImage, setDisplayImage ] = useState(Images.Default);

    // Defining effects
    // High usage effect
    useEffect(() => {
        setHighUsage(GetAverageUsage(overallData, 7, 2));
    }, [overallData]);

    // Image effect
    useEffect(() => {
        // Returning on invalid data
        if (highUsage === 0.0 || dailyData.length <= 1)
        {
            return;
        }
        
        // Calculating new image based on quartile increments
        const percentUsage = dailyData[dailyData.length - 1][1] / highUsage
        if (percentUsage <= 0.25)
        {
            setDisplayImage(Images.Forest1);
        }
        else if (percentUsage <= 0.5)
        {
            setDisplayImage(Images.Forest2);
        }
        else if (percentUsage <= 0.75)
        {
            setDisplayImage(Images.Forest3);
        }
        else
        {
            setDisplayImage(Images.Forest4);
        }
    }, [dailyData, highUsage]);

    return(
        <div className="Slideshow">
            <Link to={"../" + PageNames.HOME_PAGE_NAME}>
                <img src={displayImage} alt="forest" />
            </Link>
        </div>
    );
}

export default Slideshow;