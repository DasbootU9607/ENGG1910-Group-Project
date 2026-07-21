# Project Plan

Project topic: AI-Powered Misinformation Risk Assistant for Social Media Users

## Course Requirement Mapping

The ENGG1910 guideline asks the written report to explain motivation, working principles, key findings, reflection, and the AI system design. The report should be under 1500 English words and may include figures, tables, and references. Based on the sample reports, a suitable structure is:

1. Title and abstract
2. Introduction and motivation
3. Main problem formulation
4. Data requirement and preparation
5. AI system architecture and algorithm design
6. Demonstration experiment and observations
7. Challenges, limitations, and reflection
8. Conclusion
9. References

## System Goal

The system should not claim that a social media post is true or false. Instead, it estimates misinformation risk and explains why the post deserves careful checking before sharing.

## Prototype Scope

The current prototype is an offline demo that can run without paid APIs or large datasets. It uses:

- Synthetic labelled social media posts for a course demonstration.
- TF-IDF text features for NLP signals.
- Numeric risk features, such as source credibility, repost speed, account age, punctuation intensity, emotional wording, and unsupported claim markers.
- Logistic regression to estimate the probability that a post belongs to the higher-risk class.
- Rule-based explanations to highlight suspicious elements.

## Inputs

Each post record contains:

- `text`: social media post text.
- `source_domain`: linked source or posting source.
- `account_age_days`: age of the account.
- `follower_count` and `following_count`: simple account credibility indicators.
- `repost_count_1h`: number of reposts in the first hour.
- `distinct_accounts_1h`: number of distinct reposting accounts.
- `has_link`: whether the post includes a link.
- `label`: demo label, where 1 means higher misinformation risk and 0 means lower risk.

## Outputs

The demo generates:

- `outputs/evaluation_metrics.json`: model metrics.
- `outputs/predictions.csv`: risk score and explanation for each post.
- `outputs/risk_distribution.png`: simple visualization for the report.

## Data and API Needs

No database or API is required for the current prototype.

For a stronger final experiment, the group can manually download one or more public datasets and map them into the same CSV format:

- LIAR dataset: short political claims with truthfulness labels.
- FakeNewsNet: news/social context data from PolitiFact and GossipCop.
- CoAID: COVID-19 healthcare misinformation dataset.
- A source reliability table, such as an internally curated list of official, mainstream, unknown, and low-credibility domains.

Commercial source credibility APIs are optional, not required. They may raise cost, licensing, and reproducibility issues.

## Work Plan

1. Build offline prototype and verify that it runs.
2. Ask group members to run the demo and collect generated output files.
3. Use the output metrics, risk examples, and chart in the written report.
4. If time allows, replace synthetic data with a public dataset and rerun the same pipeline.
