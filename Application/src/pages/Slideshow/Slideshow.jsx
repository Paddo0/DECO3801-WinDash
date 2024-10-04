import { useCallback, useContext, useEffect, useState } from "react";
import { SettingsContext } from "../Settings/SettingsContext";
import Images from "../../utils/SlideshowImport";
import { PageNames } from "../../data/constants";
import { Outlet, Link } from "react-router-dom";
import { db } from "../../firebase";
import { collection, getDocs, query, where, documentId } from "firebase/firestore";

/**
 * Base slideshow page component
 * @returns {React.JSX.Element} Slideshow component
 */
function Slideshow() {
    // Settings Config
    const { config } = useContext(SettingsContext);

    useEffect(() => {
        const getDailyData = async () => {
            var dailyDataCollection = collection(db, "dailyData");
            dailyDataCollection = query(dailyDataCollection, where(documentId(), '==', config.meterId))
            const dailyDataSnapshot = await getDocs(dailyDataCollection);
            const dailyDataList = dailyDataSnapshot.docs.map(doc => doc.data());
            console.log(dailyDataList[0]["seriesData"][0]["Voltage"]);
            setTestOutput(dailyDataList[0]["seriesData"][0]["Voltage"]);
            return dailyDataList;
        };
        getDailyData();
    }
    , [config.meterId]);
        
    //const TestGetData = useCallback();

    const [ testOutput, setTestOutput] = useState("e");

    return(
        <>{testOutput}</>
        //<div className="Slideshow">
        //    <Link to={"../" + PageNames.HOME_PAGE_NAME}>
        //        <img src={Images.Forest1} alt="forest" />
        //    </Link>
        //</div>
    );
}

export default Slideshow;