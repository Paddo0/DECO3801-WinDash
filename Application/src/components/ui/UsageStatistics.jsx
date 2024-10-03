import React from 'react';

/**
 * Defines the statistics display component to display statistics to users
 * @returns {React.JSX.Element} Component containing statistics component
 */
function UsageStatistics({ title, summaryData }) {
    // Function to generate summary labels
    const SummaryLabels = summaryData.map((item, index) => (
        <p key={index}>{item[0]}</p>
    ));

    // Function to generate summary values and progress bars
    const SummaryValues = summaryData.map((item, index) => {
        const [label, value, unit, maxValue] = item;
        const percentage = maxValue ? ((value / maxValue) * 100).toFixed(1) : 0;

        return (
            <div key={index}>
                <p>{value} {unit}</p>
                {maxValue ? (
                    <>
                        <progress value={value} max={maxValue}></progress>
                        <span>{percentage}%</span>
                    </>
                ) : (
                    <span>No limit</span>
                )}
            </div>
        );
    });

    return (
        <div className="UsageStatistics">
            <div className="UsageStatisticsBar">
                <p>{title}</p>
            </div>
            
            <div className="UsageStatisticsDisplay">
                <div className="UsageStatisticsLabels">
                    {SummaryLabels}
                </div>

                <div className="UsageStatisticsValues">
                    {SummaryValues}
                </div>
            </div>
        </div>
    );
}

export default UsageStatistics;
