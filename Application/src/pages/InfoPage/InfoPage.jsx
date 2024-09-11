import React from 'react';
import '../../assets/styles/components/infoPage.css'; // Custom styling

function InfoPage()
{
  return (
    <div className="info-page">
      <header className="page-header">
        <h1>Help / Info</h1>
      </header>
      <main className="content">
        <section id="setup">
          <h2>Meter Setup Information</h2>
          <p>
            To set up your meter, first unpack all the components from the box. Ensure that you have the meter device, the power cable, and the user manual. Follow these steps:
          </p>
          <ol>
            <li>Connect the meter to a power source using the provided cable.</li>
            <li>Download the companion app from our website or mobile store.</li>
            <li>Follow the in-app setup guide to connect your meter to the app via Bluetooth.</li>
            <li>Calibrate your meter according to the instructions provided in the app.</li>
          </ol>
          <p>If you encounter any issues, please refer to the troubleshooting section below.</p>
          {/* Placeholder for the diagram */}
          <div className="diagram-placeholder">[Meter Diagram]</div>
        </section>

        <section id="usage">
          <h2>Usage Limits Information</h2>
          <p>
            It is important to understand the usage limits of your meter to ensure its longevity and accuracy:
          </p>
          <ul>
            <li>Do not exceed the maximum daily usage of 12 hours.</li>
            <li>Avoid exposing the meter to extreme temperatures or moisture.</li>
            <li>Regularly clean the device according to the guidelines in the user manual.</li>
          </ul>
        </section>

        <section id="faq">
          <h2>Frequently Asked Questions</h2>
          <div className="faq-item">
            <h3>1. Do you offer a free trial?</h3>
            <p>Yes, a 14-day free trial is available for all plans.</p>
          </div>
          <div className="faq-item">
            <h3>2. Do I need a credit card to start?</h3>
            <p>No, you can sign up and explore the basic features without needing a credit card.</p>
          </div>
          {/* Additional FAQs can be added similarly */}
        </section>

        <section id="contact">
          <h2>Contact Support</h2>
          <p>If you need further assistance, reach out to our support team via email at support@ourcompany.com or call us at 1-800-123-4567.</p>
        </section>
      </main>
    </div>
  );
};

export default InfoPage;
