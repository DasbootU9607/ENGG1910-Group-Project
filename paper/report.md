# AI-Powered Misinformation Risk Assistance for Social Media Users

## Abstract

We present a misinformation risk assistant for social media users. The system does not make a final true-or-false judgment; instead, it estimates whether a short claim deserves further verification and returns an interpretable risk level. We evaluate the idea on LIAR, a public benchmark of 12.8K political claims labelled by professional fact-checkers. The six LIAR labels are mapped into a binary risk task, where false, barely-true, and pants-fire claims are treated as higher risk. Our strongest model combines TF-IDF text features, lightweight metadata features, and logistic regression, reaching 0.743 accuracy, 0.712 F1, and 0.831 ROC-AUC on the official test set. A compact BERT-style transformer fine-tuned under CPU-limited settings reaches 0.609 accuracy and 0.631 ROC-AUC. The comparison shows that model complexity alone is not sufficient; data scale, tuning, and interpretability matter for risk-sensitive user-facing AI.

## 1. Introduction

Social media platforms allow misleading claims to spread before users can verify them. A direct automated truth detector is dangerous because many posts are incomplete, ambiguous, contextual, or developing. We therefore formulate the application as risk assistance rather than truth adjudication. Given a post and available context, the system estimates misinformation risk and explains warning signals so that users can decide whether to check official sources before sharing.

Our contributions are threefold. First, we design a modular risk-assistant architecture that combines linguistic signals, metadata, supervised learning, and explanation. Second, we implement two reproducible classifiers: a transparent TF-IDF logistic regression baseline and a compact transformer model. Third, we analyze performance, error patterns, and limitations on the LIAR benchmark.

## 2. Task and Data

We use the LIAR dataset, which contains short political statements collected from PolitiFact with six truthfulness labels. We map the original labels into a binary risk task. False, barely-true, and pants-fire are assigned to the higher-risk class, while half-true, mostly-true, and true are assigned to the lower-risk class. This mapping intentionally avoids claiming that all lower-risk statements are fully true; it only separates claims that are more likely to need urgent verification.

The training and validation splits are combined for training, giving 11,553 examples. The official test split contains 1,283 examples. Besides the statement text, LIAR includes subject, context, speaker identity, party affiliation, and speaker fact-checking history. Our prototype uses the statement, subject, context, and speaker history.

## 3. System Architecture

The application has four stages. The input stage receives a social-media-style claim, link/source information, account metadata, and repost metadata when available. The feature stage converts the input into numeric representations. The baseline uses TF-IDF unigrams and bigrams, plus engineered metadata such as the speaker's historical high-risk ratio. The transformer path tokenizes the claim and fine-tunes a compact BERT classifier.

The learning stage outputs a probability score between 0 and 1. Scores below 0.40 are shown as Low risk, scores between 0.40 and 0.70 as Medium risk, and scores above 0.70 as High risk. The explanation stage reports interpretable warnings, such as emotional wording, unsupported claim markers, suspicious source information, fast reposting, or a speaker with a high-risk fact-checking history. In the LIAR experiment, network features are unavailable, so explanations rely mostly on text and speaker-history signals.

## 4. Models

The first model is a logistic regression classifier over sparse TF-IDF vectors and metadata features. This model is strong for short-text classification because it captures discriminative phrases and remains easy to inspect.

The second model is a compact BERT-style transformer, `prajjwal1/bert-tiny`, fine-tuned for binary classification. Transformers use self-attention to model interactions between words and can in principle capture richer semantics than TF-IDF. We use a small checkpoint because the experiment is designed to run on CPU in a course setting.

## 5. Results

The TF-IDF logistic regression model performs best in our experiments, with 0.743 accuracy, 0.692 precision, 0.732 recall, 0.712 F1, and 0.831 ROC-AUC. The confusion matrix contains 546 true negatives, 181 false positives, 149 false negatives, and 407 true positives.

The transformer model is weaker under our limited compute setting: 0.609 accuracy, 0.574 precision, 0.378 recall, 0.456 F1, and 0.631 ROC-AUC after five epochs on 2,000 sampled training examples. The result does not mean transformers are unsuitable; rather, it shows that small transformers need enough data, tuning, and compute to outperform a strong sparse baseline.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.743 | 0.692 | 0.732 | 0.712 | 0.831 |
| BERT-tiny Transformer | 0.609 | 0.574 | 0.378 | 0.456 | 0.631 |

## 6. Analysis

The label distribution shows that the LIAR test set is not dominated by a single class. The risk distribution also shows that the assistant does not simply label everything as high risk: it predicts 578 Low, 349 Medium, and 356 High risk cases. This is useful for the intended application because users receive prioritization rather than hard moderation.

The baseline ROC curve is much stronger than the transformer curve. The baseline also makes fewer false negatives, which is important because missing a high-risk claim is costly in this application. The transformer produces many false negatives, suggesting that the small model underfits with the current training budget.

## 7. Ethical Considerations and Limitations

The system can support critical thinking, but it should not be used as an automatic censorship tool. False positives may unfairly reduce trust in legitimate speech, while false negatives may give users false confidence. Political fact-checking datasets can encode sampling bias because they focus on particular speakers, topics, and media environments. Privacy is also important: a deployed system should minimize account and repost data collection. Finally, adversarial users may rewrite claims to avoid obvious warning signals.

## 8. Conclusion

We built a reproducible misinformation risk assistant using LIAR. The best current model is a transparent TF-IDF logistic regression classifier with metadata features, while a compact transformer provides a deep learning comparison. The project demonstrates that an effective AI application is not only about choosing a powerful model; it also requires careful task framing, data preparation, explanation, and ethical boundaries. Future work should test larger transformers, multilingual claims, source-reliability databases, and real repost-network features.

## References

[1] W. Y. Wang. 2017. Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection. Proceedings of ACL.

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.

[3] Hugging Face. ucsbnlp/liar Dataset Repository. https://huggingface.co/datasets/ucsbnlp/liar

[4] PolitiFact. Fact-checking political claims. https://www.politifact.com/
