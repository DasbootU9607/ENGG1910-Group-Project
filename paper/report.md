# AI-Powered Misinformation Risk Assistance for Social Media Users

## Abstract

We present a misinformation risk assistant for social media users. The system does not make a final true-or-false judgment; instead, it estimates whether a short claim deserves further verification and returns an interpretable risk level. We evaluate the idea on LIAR, a public benchmark of 12.8K political claims labelled by professional fact-checkers. The six LIAR labels are mapped into a binary risk task, where false, barely-true, and pants-fire claims are treated as higher risk. Under a fair text-only protocol, BERT-base reaches 0.623 F1 and 0.697 ROC-AUC, slightly ahead of a text-only TF-IDF logistic regression baseline at 0.617 F1 and 0.658 ROC-AUC. A metadata-enhanced TF-IDF model reaches 0.719 F1 and 0.830 ROC-AUC, showing the value of speaker-history information when it is available.

## 1. Introduction

Social media platforms allow misleading claims to spread before users can verify them. A direct automated truth detector is dangerous because many posts are incomplete, ambiguous, contextual, or developing. We therefore formulate the application as risk assistance rather than truth adjudication. Given a post and available context, the system estimates misinformation risk and explains warning signals so that users can decide whether to check official sources before sharing.

Our contributions are threefold. First, we design a modular risk-assistant architecture that combines linguistic signals, metadata, supervised learning, and explanation. Second, we implement reproducible TF-IDF and BERT-base classifiers under a fair evaluation protocol. Third, we analyze performance, error patterns, and limitations on the LIAR benchmark.

## 2. Task and Data

We use the LIAR dataset, which contains short political statements collected from PolitiFact with six truthfulness labels. We map the original labels into a binary risk task. False, barely-true, and pants-fire are assigned to the higher-risk class, while half-true, mostly-true, and true are assigned to the lower-risk class. This mapping intentionally avoids claiming that all lower-risk statements are fully true; it only separates claims that are more likely to need urgent verification.

The fair experiment uses the official LIAR training split for model fitting, the official validation split for threshold selection and transformer checkpoint selection, and the official test split for final reporting. This gives 10,269 training examples, 1,284 validation examples, and 1,283 test examples. Besides the statement text, LIAR includes subject, context, speaker identity, party affiliation, and speaker fact-checking history. We report text-only models separately from the metadata-enhanced model because the latter uses extra speaker-history information.

## 3. System Architecture

The application has four stages. The input stage receives a social-media-style claim, link/source information, account metadata, and repost metadata when available. The feature stage converts the input into numeric representations. The baseline uses TF-IDF unigrams and bigrams, plus engineered metadata such as the speaker's historical high-risk ratio. The transformer path tokenizes the claim and fine-tunes a compact BERT classifier.

The learning stage outputs a probability score between 0 and 1. Scores below 0.40 are shown as Low risk, scores between 0.40 and 0.70 as Medium risk, and scores above 0.70 as High risk. The explanation stage reports interpretable warnings, such as emotional wording, unsupported claim markers, suspicious source information, fast reposting, or a speaker with a high-risk fact-checking history. In the LIAR experiment, network features are unavailable, so explanations rely mostly on text and speaker-history signals.

## 4. Models

The first fair baseline is a logistic regression classifier over sparse TF-IDF text vectors. This model is strong for short-text classification because it captures discriminative phrases and remains easy to inspect. We also train a metadata-enhanced variant that adds speaker-history and lightweight risk features.

The second fair model is `bert-base-uncased`, fine-tuned for binary classification on the same text-only inputs. Transformers use self-attention to model interactions between words and can capture richer semantics than TF-IDF. We train for up to five epochs on GPU and select the checkpoint with the best validation F1.

## 5. Results

Under the fair text-only protocol, BERT-base slightly outperforms the text-only TF-IDF model by F1 and ROC-AUC. BERT-base reaches 0.504 accuracy, 0.464 precision, 0.948 recall, 0.623 F1, and 0.697 ROC-AUC. Text-only TF-IDF reaches 0.509 accuracy, 0.466 precision, 0.912 recall, 0.617 F1, and 0.658 ROC-AUC.

The metadata-enhanced TF-IDF model performs best overall, with 0.704 accuracy, 0.610 precision, 0.876 recall, 0.719 F1, and 0.830 ROC-AUC. This model is not a direct fair comparison against BERT-base because it uses extra speaker-history metadata. Its result shows that contextual metadata is highly informative for this risk-assistance task.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF text-only | 0.509 | 0.466 | 0.912 | 0.617 | 0.658 |
| BERT-base text-only | 0.504 | 0.464 | 0.948 | 0.623 | 0.697 |
| TF-IDF + metadata | 0.704 | 0.610 | 0.876 | 0.719 | 0.830 |

## 6. Analysis

The label distribution shows that the LIAR test set is not dominated by a single class. The selected text-only models favor high recall because the validation threshold is optimized for F1 and missing a higher-risk claim is costly in the intended application.

The ROC curves show that BERT-base separates text-only examples better than TF-IDF text-only, while metadata-enhanced TF-IDF is strongest when speaker-history information is allowed. BERT-base's best checkpoint occurs at epoch 2; later epochs reduce validation F1 despite lower training loss, indicating overfitting.

## 7. Ethical Considerations and Limitations

The system can support critical thinking, but it should not be used as an automatic censorship tool. False positives may unfairly reduce trust in legitimate speech, while false negatives may give users false confidence. Political fact-checking datasets can encode sampling bias because they focus on particular speakers, topics, and media environments. Privacy is also important: a deployed system should minimize account and repost data collection. Finally, adversarial users may rewrite claims to avoid obvious warning signals.

## 8. Conclusion

We built a reproducible misinformation risk assistant using LIAR. Under a fair text-only comparison, BERT-base slightly outperforms TF-IDF logistic regression, while the metadata-enhanced TF-IDF model remains strongest overall because it uses additional speaker-history information. The project demonstrates that an effective AI application requires careful task framing, data preparation, feature access, explanation, and ethical boundaries. Future work should test multilingual claims, source-reliability databases, and real repost-network features.

## References

[1] W. Y. Wang. 2017. Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection. Proceedings of ACL.

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.

[3] Hugging Face. ucsbnlp/liar Dataset Repository. https://huggingface.co/datasets/ucsbnlp/liar

[4] PolitiFact. Fact-checking political claims. https://www.politifact.com/
