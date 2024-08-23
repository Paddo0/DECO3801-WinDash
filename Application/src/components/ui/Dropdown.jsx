

/**
 * Defines the graph dropdown component to allow different display configurations
 * @returns {React.JSX.Element} Div containing dropdown menu
 */
function Dropdown(props)
{
    return (
        <div className="Dropdown">
            <select placeholder={props.placeholder}>

                <option value="1st">1st Configuration</option>

                <option value="2nd">2nd Configuration</option>

                <option value="3rd">3rd Configuration</option>

            </select>
        </div>
    )
}

export default Dropdown