def generate_recommendations(customer_data, probability, drivers):
    """
    Generate business-oriented retention recommendations
    based on churn probability and customer attributes.
    """

    recommendations = []

    # -----------------------------
    # Risk priority
    # -----------------------------
    if probability >= 0.60:
        priority = "Critical"

    elif probability >= 0.50:
        priority = "High"

    elif probability >= 0.25:
        priority = "Medium"

    else:
        priority = "Low"

    # -----------------------------
    # Contract
    # -----------------------------
    if customer_data.get("Contract") == "Month-to-month":
        recommendations.append({
            "category": "Contract",
            "action": "Offer a discounted one-year or two-year contract.",
            "reason": "Month-to-month customers generally have lower contractual commitment."
        })

    # -----------------------------
    # Tech support
    # -----------------------------
    if customer_data.get("TechSupport") == "No":
        recommendations.append({
            "category": "Support",
            "action": "Offer technical support or a complimentary support trial.",
            "reason": "Providing support can improve customer experience and retention."
        })

    # -----------------------------
    # Online security
    # -----------------------------
    if customer_data.get("OnlineSecurity") == "No":
        recommendations.append({
            "category": "Security",
            "action": "Offer an online security add-on or bundled security plan.",
            "reason": "Additional services can increase product value and engagement."
        })

    # -----------------------------
    # Payment method
    # -----------------------------
    if customer_data.get("PaymentMethod") == "Electronic check":
        recommendations.append({
            "category": "Payment",
            "action": "Encourage automatic payment enrollment with an incentive.",
            "reason": "Automatic payment can improve payment continuity."
        })

    # -----------------------------
    # Tenure
    # -----------------------------
    tenure = customer_data.get("tenure", 0)

    if tenure <= 12:
        recommendations.append({
            "category": "Early Retention",
            "action": "Launch an early-tenure retention campaign.",
            "reason": "The customer has relatively short tenure."
        })

    # -----------------------------
    # Monthly charges
    # -----------------------------
    monthly_charges = customer_data.get("MonthlyCharges", 0)

    if monthly_charges >= 80:
        recommendations.append({
            "category": "Pricing",
            "action": "Review the customer's current plan and offer a suitable bundled package.",
            "reason": "The customer's monthly charges are relatively high."
        })

    # -----------------------------
    # Behavioral Features
    # -----------------------------
    usage_change = customer_data.get("usage_change_30d", 0)
    if usage_change < 0:
        recommendations.append({
            "category": "Engagement",
            "action": "Send a targeted re-engagement email or customized content.",
            "reason": f"Customer's usage has dropped by {abs(usage_change):.1f}% over the last 30 days."
        })

    support_tickets = customer_data.get("support_tickets_30d", 0)
    if support_tickets >= 3:
        recommendations.append({
            "category": "Support",
            "action": "Route immediately to a premium white-glove technical support agent.",
            "reason": f"Customer has filed {support_tickets} support tickets in the last month."
        })

    active_days = customer_data.get("active_days_30d", 30)
    if active_days <= 10:
        recommendations.append({
            "category": "Retention",
            "action": "Trigger a 'we miss you' discount campaign.",
            "reason": f"Customer was only active for {active_days} out of the last 30 days."
        })

    # -----------------------------
    # High-risk escalation
    # -----------------------------
    if probability >= 0.60:
        recommendations.insert(0, {
            "category": "Priority",
            "action": "Prioritize this customer for immediate retention outreach.",
            "reason": f"Predicted churn probability is {probability * 100:.1f}%."
        })

    elif probability >= 0.50:
        recommendations.insert(0, {
            "category": "Priority",
            "action": "Include this customer in a targeted retention campaign.",
            "reason": f"Predicted churn probability is {probability * 100:.1f}%."
        })

    # -----------------------------
    # Low-risk customer
    # -----------------------------
    if probability < 0.25:
        recommendations.append({
            "category": "Engagement",
            "action": "Continue normal customer engagement and monitor future risk.",
            "reason": "Current predicted churn probability is relatively low."
        })

    return {
        "priority": priority,
        "actions": recommendations
    }