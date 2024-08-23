import Dropdown from "./Dropdown";
import PlaceholderGraph from "../../assets/images/debug-graph-placeholder.PNG";


/**
 * Defines the graph display component to display usage data
 * @returns {React.JSX.Element} Component containing graph component
 */
function UsageGraph(props)
{
    const options = [
        'one', 'two', 'three'
      ];
    return(
        <div className="UsageGraph">
            <div className="UsageGraphBar">
                <div className='UsageGraphControl'>
                    <p>{props.title}</p>
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown placeholder="Select Option" />
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown placeholder="Select Option" />
                </div>

                <div className='UsageGraphControl'>
                    <Dropdown placeholder="Select Option" />
                </div>
            </div>

            <div className="UsageGraphDisplay">
                <img src={PlaceholderGraph} alt="graph" />
            </div>

        </div>
    )
}

export default UsageGraph;