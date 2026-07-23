# AI-Powered Misinformation Risk Assistance for Social Media Users

## Abstract

This project designs an AI-powered assistant that helps social media users decide whether a short online claim deserves further checking before it is shared. The system is not designed as an automatic truth judge. Instead, it predicts a three-level misinformation risk label, converts the output into Low, Medium, or High warning language, and gives simple reasons for the warning. Our experiment uses the LIAR dataset, a public benchmark of 12.8K political claims labelled by professional fact-checkers. The original six labels are mapped into a three-level risk task: true and mostly-true are treated as Low risk, half-true and barely-true are treated as Medium risk, and false and pants-fire are treated as High risk. The baseline is a TF-IDF logistic regression model trained only on claim text, which obtains 0.421 macro-F1 and 0.611 one-vs-rest macro ROC-AUC. The main experimental model is BERT-base fine-tuned on the same text-only input, which improves performance to 0.491 macro-F1 and 0.670 macro ROC-AUC.

## I. Introduction and Motivation

Social media allows information to spread quickly, but it also allows misleading claims to reach many users before they have time to verify them. A user may see a claim in a repost, screenshot, short video caption, or political quote, and the decision to share is often made within seconds. This creates a practical need for an AI tool that can slow down risky sharing and encourage users to check reliable sources.

However, a direct "true or false" detector is not suitable for this problem. Online claims are often incomplete, emotional, or dependent on context. Some claims are still developing, and some may be partly true but misleading. Therefore, our project frames the task as risk assistance. The system warns the user that a claim may require verification, but it does not censor the claim or replace human judgment.

The goal is to build and evaluate a reproducible prototype. It receives a social-media-style claim, extracts text and context features, predicts a risk level, and returns an interpretable warning. This matches the course project requirement because it includes a real AI application scenario, data preparation, model design, evaluation, and reflection.

## II. Main Problem

The main problem is to estimate whether a short claim belongs to Low, Medium, or High misinformation risk. There are three difficulties. First, the input text is short, so the model has limited evidence. Second, social media posts may need extra information such as the source, author history, and repost pattern, but this information is not always available. Third, an incorrect warning has consequences: false positives may reduce trust in legitimate speech, while false negatives may give users false confidence.

For this reason, the model output is treated as a warning signal instead of a final decision. In the application design, a High risk label means "please verify this claim before sharing", not "this claim is certainly false".

## III. Data Requirement and Preparation

A deployed version would require the claim text, link or source information, account information, repost speed, number of distinct reposting accounts, and optionally source-reliability data. These features help detect emotional wording, unsupported claims, weak sources, fast reposting, and unusual account behavior.

For the reproducible experiment, we use the LIAR dataset. LIAR contains short political statements from PolitiFact with six labels: pants-fire, false, barely-true, half-true, mostly-true, and true. We convert them into a three-level risk task. The Low risk class contains true and mostly-true. The Medium risk class contains half-true and barely-true. The High risk class contains false and pants-fire. This mapping is more informative than the earlier two-class design because partly true and ambiguous claims are not forced into either the safest or riskiest group.

The experiment follows the official LIAR split. There are 10,269 training examples, 1,284 validation examples, and 1,283 test examples. The text input combines the statement, subject, and context fields. The validation split is used for transformer checkpoint selection. The test split is used only once for final reporting.

## IV. Application Design and AI Method

The proposed system has four stages. The input stage receives a post, source information, account information, and repost information when available. The feature stage transforms the input into numerical features. Text is represented by TF-IDF unigrams and bigrams for the baseline, while the transformer path tokenizes the claim with BERT-base.

The learning stage predicts probabilities for Low, Medium, and High risk. The displayed risk score is computed as `0.5 * prob_medium + prob_high`, so Medium risk contributes partial concern and High risk contributes full concern. For the formal LIAR experiment, predictions are selected by multiclass argmax. We report macro precision, macro recall, macro-F1, weighted-F1, and multiclass one-vs-rest macro ROC-AUC.

The explanation stage translates model signals into short user-facing reasons. Examples include emotional wording, unsupported claim markers, weak or unknown source, fast reposting, or heavy punctuation. These explanations make the system more useful than a black-box label because the user can see what should be checked.

## V. Experimental Design

We evaluate two models. The first model is a text-only TF-IDF logistic regression classifier. It is a strong baseline for short-text classification and is easy to reproduce. The second model is BERT-base fine-tuned on the same text-only inputs. This is the fair comparison with TF-IDF because both models receive the same information.

BERT-base is trained for up to five epochs on GPU with batch size 8, maximum sequence length 128, learning rate 2e-5, and balanced class weights. After each epoch, the model is evaluated on the validation set. The checkpoint with the best validation macro-F1 is selected; in our run, this occurs at epoch 2. We report all final metrics on the official test split.

## VI. Key Findings and Observations

The text-only comparison shows that BERT-base performs clearly better than TF-IDF on the three-class task. BERT-base reaches 0.496 accuracy, 0.495 macro precision, 0.489 macro recall, 0.491 macro-F1, 0.495 weighted-F1, and 0.670 macro ROC-AUC. TF-IDF text-only reaches 0.422 accuracy, 0.424 macro precision, 0.427 macro recall, 0.421 macro-F1, 0.421 weighted-F1, and 0.611 macro ROC-AUC. This suggests that contextual representation helps when the model must separate Low, Medium, and High risk instead of only making a two-class warning decision.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF text-only | 0.422 | 0.424 | 0.427 | 0.421 | 0.421 | 0.611 |
| BERT-base text-only | 0.496 | 0.495 | 0.489 | 0.491 | 0.495 | 0.670 |

The three-class setting is more difficult than the earlier two-class setting because ambiguous claims are kept as their own Medium risk category. The confusion matrices show that Medium risk is the hardest class to separate, which is expected because half-true and barely-true statements often contain both accurate and misleading elements. For a warning assistant, this behavior is useful: Medium risk gives the interface a way to ask for caution without treating every uncertain claim as High risk.

## VII. Discussion and Reflection

This project shows that the hardest part of an AI application is not only choosing a model. The task framing, data access, evaluation protocol, and ethical boundary are equally important. If the system is described as a truth detector, the same model could be misleading or harmful. If it is described as a risk assistant, the output becomes a useful prompt for careful checking.

There are several limitations. LIAR focuses on political fact-checking in the United States, so the model may not generalize to health, finance, entertainment, or multilingual claims. In addition, fact-checking datasets can reflect sampling bias because fact-checkers choose which claims to investigate.

Future work should test multilingual social media posts, add source-reliability databases, and use real repost-network features. The system should also be evaluated with human users to see whether the warning reduces careless sharing without causing excessive distrust.

## VIII. Summary

We built a reproducible misinformation risk assistant and tested it on the LIAR benchmark. The final version uses a three-level LIAR mapping: Low risk for true and mostly-true, Medium risk for half-true and barely-true, and High risk for false and pants-fire. Under a fair text-only protocol, BERT-base outperforms TF-IDF logistic regression. Overall, the project demonstrates how AI can support user judgment when the application is carefully framed, transparently evaluated, and ethically limited.

## References

[1] W. Y. Wang. 2017. Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection. Proceedings of ACL.

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.

[3] Hugging Face. ucsbnlp/liar Dataset Repository. https://huggingface.co/datasets/ucsbnlp/liar

[4] PolitiFact. Fact-checking political claims. https://www.politifact.com/
