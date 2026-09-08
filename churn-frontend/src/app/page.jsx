"use client";

import React, { useMemo, useState } from "react";

const workflowData = [
  ["01 · Customer data","Start with the customer story.","Profile, tenure, contract and service attributes become the input to the prediction layer."],
  ["02 · ML prediction","Turn customer attributes into risk.","The selected Random Forest model produces a probability score that can be consumed by the product layer."],
  ["03 · SHAP analysis","Explain why the score changed.","Feature-level contributions surface the strongest risk drivers instead of leaving the prediction as a black box."],
  ["04 · Retention action","Convert risk into a next step.","Prioritize the customer and recommend an intervention based on the strongest signals."]
];

const defaults = { tenure: 65, charges: 29, contract: "monthly" };

function scoreCustomer(tenure, charges, contract) {
  let risk = 0.22;
  risk += Math.max(0, (48 - tenure) / 48) * 0.25;
  risk += Math.max(0, (charges - 55) / 65) * 0.12;
  risk += contract === "monthly" ? 0.28 : contract === "one" ? 0.10 : -0.08;
  return Math.max(0.03, Math.min(0.97, risk));
}

function getRiskClass(risk) {
  return risk < 0.40 ? "low" : risk < 0.70 ? "medium" : "high";
}

function getRiskLabel(risk) {
  return risk < 0.40 ? "Low risk" : risk < 0.70 ? "Medium risk" : "High risk";
}

function getRecommendation(risk) {
  if (risk >= 0.70) return [
    "Prioritize this customer.",
    "Offer a contract incentive and technical support option to reduce churn risk."
  ];
  if (risk >= 0.40) return [
    "Monitor this customer.",
    "Consider proactive outreach and a service review before risk increases."
  ];
  return [
    "Maintain the relationship.",
    "Customer signals are relatively stable. Continue engagement and monitor changes."
  ];
}

export default function Home() {
  const [workflowStep, setWorkflowStep] = useState(0);
  const [appTab, setAppTab] = useState("overview");
  const [tenure, setTenure] = useState(defaults.tenure);
  const [charges, setCharges] = useState(defaults.charges);
  const [contract, setContract] = useState(defaults.contract);
  const [toast, setToast] = useState("");
  const [modalOpen, setModalOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [apiPrediction, setApiPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  const prediction = useMemo(() => {
    const risk = scoreCustomer(tenure, charges, contract);
    const rc = getRiskClass(risk);
    const contractImpact = contract === "monthly" ? 0.28 : contract === "one" ? 0.12 : 0.05;
    const tenureImpact = Math.max(0.02, ((48 - tenure) / 48) * 0.24);
    const chargesImpact = Math.max(0.02, (Math.max(0, charges - 35) / 85) * 0.18);
    const supportImpact = risk > 0.70 ? 0.11 : risk > 0.40 ? 0.07 : 0.03;
    return {
      risk, rc, pct: (risk * 100).toFixed(1),
      label: getRiskLabel(risk),
      contractImpact, tenureImpact, chargesImpact, supportImpact,
      recommendation: getRecommendation(risk)
    };
  }, [tenure, charges, contract]);

  const displayedPrediction = apiPrediction
    ? {
        pct: Number(apiPrediction.churn_percentage).toFixed(1),
        label: `${apiPrediction.risk_level} risk`,
        rc: apiPrediction.risk_level.toLowerCase(),
        risk: Number(apiPrediction.churn_probability),
      }
    : prediction;
  
  const displayedRecommendation = apiPrediction?.recommendations?.actions
    ?.length
    ? {
        title: apiPrediction.recommendations.actions[0].action,
        text: apiPrediction.recommendations.actions
          .slice(0, 3)
          .map((item) => item.reason)
          .join(" "),
      }
    : {
        title: prediction.recommendation[0],
        text: prediction.recommendation[1],
      };
  
  function notify(message) {
    setToast(message);
    window.clearTimeout(window.__churnToast);
    window.__churnToast = window.setTimeout(() => setToast(""), 2400);
  }

  function scrollTo(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setMenuOpen(false);
  }

  function buildCustomerPayload() {
    return {
        gender: "Male",
        SeniorCitizen: 0,
        Partner: "Yes",
        Dependents: "No",

        tenure: Number(tenure),

        PhoneService: "Yes",
        MultipleLines: "No",
        OnlineSecurity: "No",
        OnlineBackup: "No",
        DeviceProtection: "No",
        TechSupport: "No",
        StreamingTV: "No",
        StreamingMovies: "No",
        PaperlessBilling: "Yes",

        MonthlyCharges: Number(charges),
        TotalCharges: Number(tenure) * Number(charges),

        Contract:
            contract === "monthly"
                ? "Month-to-month"
                : contract === "one"
                ? "One year"
                : "Two year",

        PaymentMethod: "Electronic check",
        InternetService: "Fiber optic"
    };
  }
  
  async function runPrediction() {
    setIsPredicting(true);
    setPredictionError(null);

    try {
        const customerData = buildCustomerPayload();
        console.log("===== REQUEST =====");
        console.log(customerData);

        const API_URL = process.env.NEXT_PUBLIC_API_URL;
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(customerData)
        });
        
        console.log("Status:", response.status);
        
        const rawResponse = await response.text();
      
        console.log("========= RESPONSE =======");
        console.log(rawResponse)

        if (!response.ok) {
          throw new Error(
            `Backend returned ${response.status}: ${rawResponse}`
          );
        }

        const result = JSON.parse(rawResponse);

        console.log("Parsed prediction:", result);

        setApiPrediction(result);

        notify("Predicted...");

    } catch (error) {
        console.log("======== PREDICTION ERROR =======")
        console.error("Prediction error:", error);
        console.error("API URL:", process.env.NEXT_PUBLIC_API_URL);
        setPredictionError(error.message);

        notify(`Prediction failed: ${error.message}`);

    } finally {
        setIsPredicting(false);
    }
}

  function resetPrediction() {
    setTenure(defaults.tenure);
    setCharges(defaults.charges);
    setContract(defaults.contract);
    setApiPrediction(null);
    setPredictionError(null);
    setAppTab("overview");
    notify("Customer prediction reset.");
  }

  function handleTenureChange(value) {
    setTenure(Number(value));
    setApiPrediction(null);
    setPredictionError(null);
  }

  function handleChargesChange(value) {
    setCharges(Number(value));
    setApiPrediction(null);
    setPredictionError(null);
  }

  function handleContractChange(value) {
    setContract(value);
    setApiPrediction(null);
    setPredictionError(null);
  }

  function openAnalytics() { setAppTab("analytics"); scrollTo("preview"); }

  const contractName = contract === "monthly" ? "Monthly" : contract === "one" ? "1 year" : "2 year";

  return (
    <>
      <div className="page-glow" />

      <header className={`nav ${menuOpen ? "nav-open" : ""}`}>
        <a className="brand" href="#product" onClick={() => setMenuOpen(false)}>
          <span className="brand-mark">C</span>
          <span>
            <strong>ChurnIQ</strong>
            <small>Retention intelligence</small>
          </span>
        </a>

        <nav className="nav-links">
          <a className="active" href="#product">
            Product
          </a>
          <a href="#intelligence">Intelligence</a>
          <a href="#workflow">How it works</a>
          <a href="#analytics">Analytics</a>
        </nav>

        <button className="nav-cta" onClick={() => scrollTo("preview")}>
          Launch platform <span>↗</span>
        </button>
        <button
          className="menu-btn"
          aria-label="Open menu"
          onClick={() => setMenuOpen((v) => !v)}
        >
          ☰
        </button>
      </header>

      <main>
        <section className="hero" id="product">
          <div className="hero-copy">
            <div className="eyebrow">
              <span /> AI-powered customer retention
            </div>
            <h1>
              Know who will churn.
              <br />
              <em>Know why.</em>
            </h1>
            <p className="hero-sub">
              Predict customer churn with machine learning, understand the
              drivers behind every prediction, and turn risk signals into
              smarter retention decisions.
            </p>
            <div className="hero-actions">
              <button
                className="primary-btn"
                onClick={() => scrollTo("preview")}
              >
                Explore the platform <span>→</span>
              </button>
              <button
                className="secondary-btn"
                onClick={() => scrollTo("workflow")}
              >
                See how it works <span>↗</span>
              </button>
            </div>
            <div className="hero-proof">
              <div>
                <strong>0.92</strong>
                <span>ROC-AUC</span>
              </div>
              <div>
                <strong>94%</strong>
                <span>Churn recall</span>
              </div>
              <div>
                <strong>7,043</strong>
                <span>Customers analyzed</span>
              </div>
            </div>
          </div>

          <div className="hero-visual">
            <div className="orbital-grid" />
            <div className="hero-orb">
              <div className="orb-core">C</div>
              <div className="orbit orbit-1" />
              <div className="orbit orbit-2" />
              <div className="signal-card card-one">
                <span>Risk score</span>
                <b>{displayedPrediction.pct}%</b>
                <small className={`status-${displayedPrediction.rc}`}>
                  {displayedPrediction.label}
                </small>
              </div>
              <div className="signal-card card-two">
                <span>Model</span>
                <b>Random Forest</b>
                <small>ROC-AUC 0.92</small>
              </div>
              <div className="signal-card card-three">
                <span>Top driver</span>
                <b>Contract</b>
                <small>Month-to-month</small>
              </div>
            </div>
          </div>
        </section>

        <section className="logo-strip">
          <span>BUILT FOR DATA-DRIVEN RETENTION</span>
          <div className="tech-list">
            <b>Python</b>
            <b>Random Forest</b>
            <b>SHAP</b>
            <b>FastAPI</b>
            <b>Next.js</b>
          </div>
        </section>

        <section className="intro section" id="intelligence">
          <div className="section-kicker">
            ONE PLATFORM · THREE INTELLIGENCE LAYERS
          </div>
          <h2>
            From customer data to
            <br />
            <em>retention action.</em>
          </h2>
          <p className="section-lead">
            ChurnIQ turns a model prediction into a clear, explainable customer
            story.
          </p>
          <div className="feature-grid">
            <article className="feature-card feature-large green-card">
              <span className="pill">01 · Predict</span>
              <h3>
                See risk before
                <br />
                customers leave.
              </h3>
              <p>
                Generate a probability-based churn score from customer profile
                and service information.
              </p>
              <div className="mini-chart">
                {[34, 48, 41, 67, 54, 82, 74, 93].map((h, i) => (
                  <i key={i} style={{ height: `${h}%` }} />
                ))}
              </div>
            </article>
            <article className="feature-card image-card">
              <div className="mock-person" />
              <div className="image-overlay">
                <span className="pill">02 · Explain</span>
                <h3>
                  Every prediction
                  <br />
                  has a reason.
                </h3>
                <p>
                  SHAP-based explanations surface the strongest drivers behind
                  individual predictions.
                </p>
              </div>
            </article>
            <article className="feature-card compact-card">
              <span className="icon-bubble">↗</span>
              <span className="pill">03 · Act</span>
              <h3>
                Turn risk into
                <br />
                retention strategy.
              </h3>
              <p>
                Translate customer risk signals into practical next-step
                recommendations.
              </p>
            </article>
            <article className="feature-card metric-card">
              <div className="metric-top">
                <span>Model performance</span>
                <span className="dot" />
              </div>
              <strong>92.3%</strong>
              <p>ROC-AUC achieved by the selected model.</p>
              <div className="metric-line">
                <span />
              </div>
            </article>
          </div>
        </section>

        <section className="workflow section" id="workflow">
          <div className="workflow-copy">
            <div className="section-kicker">HOW THE INTELLIGENCE WORKS</div>
            <h2>
              A complete ML story,
              <br />
              <em>not just a score.</em>
            </h2>
            <p>
              The experience connects prediction, explainability and business
              action into one simple flow.
            </p>
          </div>
          <div className="pipeline">
            {workflowData.map((item, i) => (
              <div className="pipeline-wrap" key={item[0]}>
                <button
                  className={`pipeline-step ${
                    workflowStep === i ? "active" : ""
                  }`}
                  onClick={() => setWorkflowStep(i)}
                >
                  <span>{item[0].slice(0, 2)}</span>
                  <b>{item[0].slice(5)}</b>
                  <small>
                    {
                      [
                        "Profile + services",
                        "RandomForest inference",
                        "Why the score changed",
                        "Prioritized response",
                      ][i]
                    }
                  </small>
                </button>
                {i < 3 && <div className="pipeline-arrow">→</div>}
              </div>
            ))}
            <div className="pipeline-detail">
              <span>{workflowData[workflowStep][0]}</span>
              <strong>{workflowData[workflowStep][1]}</strong>
              <p>{workflowData[workflowStep][2]}</p>
            </div>
          </div>
        </section>

        <section className="product-preview section" id="preview">
          <div className="preview-heading">
            <div>
              <div className="section-kicker">PRODUCT EXPERIENCE</div>
              <h2>
                Risk intelligence at
                <br />
                <em>a glance.</em>
              </h2>
            </div>
            <p>
              Designed to make complex ML outputs understandable for business
              users.
            </p>
          </div>

          <div className="app-window">
            <div className="app-top">
              <div className="app-brand">
                <span className="brand-mark small">C</span> ChurnIQ
              </div>
              <div className="app-status">
                <span /> Model online
              </div>
            </div>
            <div className="app-body">
              <aside className="app-side">
                {[
                  ["overview", "Overview"],
                  ["predict", "Predict churn"],
                  ["analytics", "Analytics"],
                  ["explain", "Explainability"],
                  ["history", "History"],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    className={appTab === id ? "side-active" : ""}
                    onClick={() => setAppTab(id)}
                  >
                    {label}
                  </button>
                ))}
              </aside>

              <div className="app-main">
                <div className="app-title">
                  <div>
                    <small>Customer risk profile</small>
                    <h3>CUST_10294</h3>
                  </div>
                  <button onClick={resetPrediction}>Reset</button>
                </div>

                <div className="app-metrics">
                  <div className={`risk-card risk-${displayedPrediction.rc}`}>
                    <small>Churn probability</small>
                    <strong
                      className={`risk-number risk-${displayedPrediction.rc}`}
                    >
                      {displayedPrediction.pct}%
                    </strong>
                    <span className={`risk-badge ${displayedPrediction.rc}`}>
                      {displayedPrediction.label}
                    </span>
                  </div>
                  <div>
                    <small>Tenure</small>
                    <strong>{tenure} mo</strong>
                    <span>
                      {tenure < 18 ? "Short tenure" : "Established customer"}
                    </span>
                  </div>
                  <div>
                    <small>Monthly charges</small>
                    <strong>${charges}</strong>
                    <span>
                      {charges > 60 ? "Above median" : "Moderate charges"}
                    </span>
                  </div>
                  <div>
                    <small>Contract</small>
                    <strong>{contractName}</strong>
                    <span>Primary driver</span>
                  </div>
                </div>

                {appTab === "overview" && (
                  <>
                    <div className="interactive-controls">
                      <div className="control">
                        <label>
                          Tenure <output>{tenure}</output> mo
                        </label>
                        <input
                          type="range"
                          min="1"
                          max="72"
                          value={tenure}
                          onChange={(e) => handleTenureChange(e.target.value)}
                        />
                      </div>
                      <div className="control">
                        <label>
                          Monthly charges <output>{charges}</output>
                        </label>
                        <input
                          type="range"
                          min="20"
                          max="120"
                          value={charges}
                          onChange={(e) => handleChargesChange(e.target.value)}
                        />
                      </div>
                      <div className="control">
                        <label>Contract</label>
                        <select
                          value={contract}
                          onChange={(e) => handleContractChange(e.target.value)}
                        >
                          <option value="monthly">Month-to-month</option>
                          <option value="one">One year</option>
                          <option value="two">Two year</option>
                        </select>
                      </div>
                      <button
                        className="predict-btn"
                        onClick={runPrediction}
                        disabled={isPredicting}
                      >
                        {" "}
                        {isPredicting ? "Predicting..." : "Run prediction →"}
                      </button>
                    </div>
                    <div className="app-lower">
                      <div className="shap-box">
                        <div className="box-head">
                          <b>Why this customer is at risk</b>
                          <span>SHAP · Random Forest</span>
                        </div>
                        {apiPrediction?.drivers?.length ? (
                          apiPrediction.drivers.map((driver, index) => (
                            <div className="bar-row" key={`${driver.feature}-${index}`}>
                              <span>{driver.display_name}</span>
                              <i>
                                <em
                                  style={{
                                    width: `${Math.min(
                                      100,
                                      Math.max(
                                        10,
                                        Math.abs(driver.impact) * 100
                                      )
                                    )}%`,
                                  }}
                                />
                              </i>
                              <b
                                className={
                                  driver.direction === "reduces_risk"
                                    ? "risk-reducing"
                                    : ""
                                }
                              >
                                {driver.direction === "reduces_risk" ? "" : "+"}
                                {Number(driver.impact).toFixed(2)}
                              </b>
                            </div>
                          ))
                        ) : (
                          <p className="shap-empty">
                            Run a prediction to view SHAP drivers
                          </p>
                        )}
                      </div>
                      <div className="recommendation">
                        <span className="pill">Retention insight</span>
                        <h4>{displayedRecommendation.title}</h4>
                        <p>{displayedRecommendation.text}</p>
                        <button onClick={() => setModalOpen(true)}>
                          View recommendation →
                        </button>
                      </div>
                    </div>
                  </>
                )}

                {appTab === "predict" && (
                  <div className="view-card">
                    <span className="pill">Prediction workspace</span>
                    <h4>Run a customer risk assessment.</h4>
                    <p>
                      Adjust the customer profile, then run the prediction. This
                      UI is ready to connect to your FastAPI endpoint.
                    </p>
                    <button
                      className="predict-btn wide"
                      onClick={runPrediction}
                      disabled={isPredicting}
                    >
                      {isPredicting ? "Predicting..." : "Run prediction →"}
                    </button>
                  </div>
                )}
                {appTab === "analytics" && (
                  <div className="view-card">
                    <span className="pill">Analytics</span>
                    <h4>Contract risk concentration</h4>
                    <div className="mini-bars">
                      <div>
                        <span>Month-to-month</span>
                        <i>
                          <em style={{ width: "88%" }} />
                        </i>
                        <b>61.2%</b>
                      </div>
                      <div>
                        <span>One year</span>
                        <i>
                          <em style={{ width: "42%" }} />
                        </i>
                        <b>24.0%</b>
                      </div>
                      <div>
                        <span>Two year</span>
                        <i>
                          <em style={{ width: "25%" }} />
                        </i>
                        <b>14.8%</b>
                      </div>
                    </div>
                  </div>
                )}
                {appTab === "explain" && (
                  <div className="view-card">
                    <span className="pill">SHAP explanation</span>
                    <h4>Why this customer is at risk.</h4>
                    {apiPrediction?.drivers?.length ? (
                      <>
                        <p>
                          The model identified the following features as
                          the strongest contributors to this customer's
                          prediction.
                        </p>
                        <div className="explain-grid">
                          {apiPrediction.drivers.map((driver, index) => (
                            <React.Fragment key={`${driver.feature}-${index}`}>
                              <span>{driver.display_name}</span>
                              <b
                                className={
                                  driver.direction === "reduces_risk"
                                    ? "risk-reducing"
                                    : ""
                                }
                              >
                                {driver.direction === "reduces_risk" ? "" : "+"}
                                {Number(driver.impact).toFixed(2)}
                              </b>
                            </React.Fragment>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p>
                        Run a prediction to view the SHAP-based explanation for
                        this customer.
                      </p>
                    )}
                  </div>
                )}
                {appTab === "history" && (
                  <div className="view-card">
                    <span className="pill">Prediction history</span>
                    <h4>Recent assessments</h4>
                    <div className="history-row">
                      <span>Today · CUST_10294</span>
                      <b>
                        {displayedPrediction.pct}% ·{" "}
                        {displayedPrediction.label.replace(" risk", "")}
                      </b>
                    </div>
                    <div className="history-row">
                      <span>Previous · CUST_10188</span>
                      <b>28.4% · Low</b>
                    </div>
                    <div className="history-row">
                      <span>Previous · CUST_10071</span>
                      <b>81.7% · High</b>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="analytics section" id="analytics">
          <div className="analytics-copy">
            <div className="section-kicker">ANALYTICS</div>
            <h2>
              See the patterns
              <br />
              behind <em>churn.</em>
            </h2>
            <p>
              Understand where risk concentrates across contracts, tenure,
              services and customer segments.
            </p>
            <button className="secondary-btn" onClick={openAnalytics}>
              Explore analytics ↗
            </button>
          </div>
          <div className="analytics-panel">
            <div className="panel-head">
              <span>Churn rate by contract</span>
              <small>Dataset · 7,043 customers</small>
            </div>
            <div className="contract-row">
              <span>Month-to-month</span>
              <i>
                <em style={{ width: "88%" }} />
              </i>
              <b>61.2%</b>
            </div>
            <div className="contract-row">
              <span>One year</span>
              <i>
                <em style={{ width: "42%" }} />
              </i>
              <b>24.0%</b>
            </div>
            <div className="contract-row">
              <span>Two year</span>
              <i>
                <em style={{ width: "25%" }} />
              </i>
              <b>14.8%</b>
            </div>
            <div className="donut">
              <span>
                <strong>26.5%</strong>
                <small>overall churn</small>
              </span>
            </div>
          </div>
        </section>

        <section className="cta section">
          <div className="cta-glow" />
          <div className="section-kicker">CUSTOMER RETENTION, REIMAGINED</div>
          <h2>
            Turn churn prediction into
            <br />
            <em>retention intelligence.</em>
          </h2>
          <p>Explore risk. Understand the why. Take action.</p>
          <button className="primary-btn" onClick={() => scrollTo("preview")}>
            Explore ChurnIQ →
          </button>
        </section>
      </main>

      {toast && (
        <div className="toast show" role="status">
          {toast}
        </div>
      )}

      {modalOpen && (
        <div
          className="modal-backdrop open"
          onClick={(e) => {
            if (e.target === e.currentTarget) setModalOpen(false);
          }}
        >
          <div className="modal">
            <button
              className="modal-close"
              aria-label="Close"
              onClick={() => setModalOpen(false)}
            >
              ×
            </button>
            <span className="pill">Retention recommendation</span>
            <h3>{displayedRecommendation.title}</h3>
            <p>
              {displayedRecommendation.text} Current estimated churn
              probability: {displayedPrediction.pct}%
            </p>
            <div className="modal-actions">
              <button
                className="primary-btn"
                onClick={() => {
                  setModalOpen(false);
                  notify("Recommendation acknowledged.");
                }}
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}

      <footer>
        <div className="brand">
          <span className="brand-mark">C</span>
          <span>
            <strong>ChurnIQ</strong>
            <small>Customer retention intelligence</small>
          </span>
        </div>
        <div className="footer-right">
          <div className="footer-links">
            <a href="https://github.com/Prabhu-Jethi" target="_blank" rel="noreferrer" className="footer-icon" aria-label="GitHub">
              <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
              </svg>
            </a>
            <a href="https://linkedin.com/in/prabhu-jethi" target="_blank" rel="noreferrer" className="footer-icon" aria-label="LinkedIn">
              <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                <rect x="2" y="9" width="4" height="12"></rect>
                <circle cx="4" cy="4" r="2"></circle>
              </svg>
            </a>
            <a href="mailto:prabhujethi9@gmail.com" className="footer-icon" aria-label="Email">
              <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                <polyline points="22,6 12,13 2,6"></polyline>
              </svg>
            </a>
          </div>
          <span>
            © 2026 ChurnIQ · Made with ❤️ by Prabhu
          </span>
        </div>
      </footer>
    </>
  );
}
