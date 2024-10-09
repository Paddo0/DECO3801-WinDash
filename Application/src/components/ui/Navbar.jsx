import Icons from "../../utils/IconImportHelper";
import { PageNames } from "../../data/constants";
import { Link, useLocation, useOutlet, useNavigate } from "react-router-dom";
import { SwitchTransition, CSSTransition } from 'react-transition-group';
import { useRef } from "react";
import { useIdleTimer } from 'react-idle-timer'

/**
 * Defines the side navigation bar that routes the user to all pages of the application
 * @returns {React.JSX.Element} Div containing navigation icons and the routed page content
 */
function Navbar()
{
    // Defining router parameters
    const location = useLocation();
    const outlet = useOutlet();
    const nodeRef = useRef(null);

    // On idle function navigate to page
    const navigate = useNavigate();
    const onIdle = () => {
      navigate("/slideshow");
    }
  
    // Defining idle timer
    useIdleTimer({
      onIdle,
      timeout: 60_000,
      throttle: 500
    });

    return (
        <>
            <div className="navbar">
                <FixedNavbarIcon icon={Icons.ArrowLeft} linkTo={"/" + PageNames.SLIDESHOW_PAGE_NAME} />
                <NavbarIcon icon={Icons.HouseIcon} linkTo={"/" + PageNames.HOME_PAGE_NAME} />
                <NavbarIcon icon={Icons.DailyIcon} linkTo={"/" + PageNames.DAILY_PAGE_NAME} />
                <NavbarIcon icon={Icons.MonthlyIcon} linkTo={"/" + PageNames.MONTHLY_PAGE_NAME} />
                <NavbarIcon icon={Icons.SettingsIcon} linkTo={"/" + PageNames.SETTINGS_PAGE_NAME} />
                <NavbarIcon icon={Icons.InfoIcon} linkTo={"/" + PageNames.INFO_PAGE_NAME} />
            </div>

            {/* Output of router content with transitions */}
            <SwitchTransition>
                <CSSTransition
                    key={location.pathname}
                    timeout={300}
                    classNames="PageContent"
                    nodeRef={nodeRef}
                >
                {() => (
                    <div className="PageContent" ref={nodeRef}>
                        {outlet}
                    </div>
                )}
                </CSSTransition>
            </SwitchTransition>
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
            <Link to={props.linkTo} >
                <div className="navbar-icon">
                    <img src={props.icon} alt="icon" />
                </div>
            </Link>
        </>
    );
}

/**
 * Defines an icon within the navbar that is fixed to the top
 * @param   {string} linkTo The page to route users to when clicked 
 * @param   {SvgFile} icon  The svg icon to be displayed in the navbar
 * @returns {React.JSX.Element} An icon component that links to other pages
 */
function FixedNavbarIcon(props)
{
    return (
        <>
            <Link to={props.linkTo}>
                <div className="fixed-navbar-icon">
                    <img src={props.icon} alt="icon" />
                </div>
            </Link>
        </>
    );
}

export default Navbar;
