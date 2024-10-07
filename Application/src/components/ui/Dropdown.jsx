

/**
 * Defines the graph dropdown component to allow different display configurations
 * @returns {React.JSX.Element} Div containing dropdown menu
 */
function Dropdown(props)
{
    const ApplyValue = (e) => {
        if (props.setValue != null)
        {
            props.setValue(props.options[e.target.value]);
        }
    }

    const CalculateDropdown = () => {
        // Returning empty component if list isn't defined
        if (props.options == null)
        {
            return (<></>);
        }
        
        // Creating dropdown with all options
        var dropdown = <></>;
        for (var i = 0; i < props.options.length; i++)
        {
            dropdown = <>
                {dropdown}
                {<option value={i}>{props.options[i].name}</option>}
            </>;
        }

        // Applying outer tags
        dropdown = (<div className="Dropdown">
            <select 
                defaultValue={props.defaultValue}
                onChange={e => ApplyValue(e)}
            >
                {dropdown}
            </select>
        </div>);

        return (dropdown);
    };

    const dropdown = CalculateDropdown();

    return (
        <>
            {dropdown}
        </>
    )
}

export default Dropdown