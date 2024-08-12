import Icons from "../../utils/IconImportHelper";
import { PageNames } from "../../data/constants";
import { Outlet, Link } from "react-router-dom";

/**
 * Defines the side navigation bar that routes the user to all pages of the application
 * @returns {React.JSX.Element} Div containing navigation icons and the routed page content
 */
function Navbar()
{
    return (
        <>
            <div className="navbar">
                <NavbarIcon icon={Icons.HouseIcon} linkTo={"/" + PageNames.HOME_PAGE_NAME} />
                <NavbarIcon icon={Icons.DailyIcon} linkTo={"/" + PageNames.DAILY_PAGE_NAME} />
                <NavbarIcon icon={Icons.MonthlyIcon} linkTo={"/" + PageNames.MONTHLY_PAGE_NAME} />
                <NavbarIcon icon={Icons.SettingsIcon} linkTo={"/" + PageNames.SETTINGS_PAGE_NAME} />
                <NavbarIcon icon={Icons.InfoIcon} linkTo={"/" + PageNames.INFO_PAGE_NAME} />

            </div>

            {/* Output of router content */}
            <div className="PageContent">
                <Outlet />
            </div>
        </>
    );
}

/**
 * Defines an icon within the navbar
 * @param   {string} linkTo The page to route users to when clicked 
 * @param   {SvgFile} icon  The svg icon to be displayed in the navbar
 * @returns {React.JSX.Element} An icon component that links to other pages
 */
function NavbarIcon(props)
{
    return (
        <>
            <Link to={props.linkTo}>
                <div className="navbar-icon">
                    <img src={props.icon} alt="icon" />
                </div>
            </Link>
        </>
    );
}

export default Navbar;
