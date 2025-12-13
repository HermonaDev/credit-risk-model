# Credit Scoring Business Understanding

## 1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

Basel II requires banks to calculate minimum capital requirements based on quantified risk. Regulatory scrutiny demands that models are transparent, justifiable, and auditable. An interpretable model allows stakeholders to validate risk assessments, ensure compliance, and explain decisions to regulators. Poor documentation or a "black-box" model risks regulatory rejection and inadequate capital allocation.

## 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks?

A proxy is necessary to transform behavioral data (RFM) into a supervised learning target. Without it, we cannot train a predictive model. The business risk is "proxy bias": the proxy (e.g., low engagement) may not perfectly correlate with actual credit default. This could lead to misclassifying good borrowers as high-risk (loss of revenue) or bad borrowers as low-risk (increased default losses).

## 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

Simple Model (Logistic Regression + WoE):
- **Pros:** Easily explained coefficients, aligned with traditional credit scoring, favorable under regulatory "right to explanation."
- **Cons:** May sacrifice predictive power if relationships are non-linear.

Complex Model (Gradient Boosting):
- **Pros:** Higher accuracy, captures complex interactions.
- **Cons:** Harder to interpret, requires extensive documentation and validation to satisfy regulators. May be viewed as a "black box."

In regulated finance, interpretability often outweighs marginal performance gains. A common compromise is to use a simple model for compliance and a complex one for benchmarking.