# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
Both models were built and compared. The rule based model was developed iteratively with targeted fixes; the ML model was trained on the same dataset to compare behavior.

**Intended purpose:**
Classify short text messages (social media style posts) into one of four mood labels: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):**
The rule based model preprocesses each post into tokens, assigns every token a sentiment weight, sums them into a score, and maps that score (plus a `mixed` check) to a label. The full scoring rules are in §3.

The ML model converts each post into a bag-of-words vector using `CountVectorizer`, then fits a `LogisticRegression` classifier on those vectors and the human-assigned labels. It learns which word combinations correlate with each label without any hand-written rules.


## 2. Data

**Dataset description:**
The dataset contains 14 short posts in `SAMPLE_POSTS`. It started with 6 provided examples; 8 more were added manually to cover a wider range of language styles including slang, emojis, sarcasm, and ambiguous moods.

**Labeling process:**
Labels were assigned by reading each post and choosing the best fit from `positive`, `negative`, `neutral`, or `mixed`. Posts with clear positive or negative tone were straightforward. Harder cases included:
- `"not sure if I'm happy or just tired 🥲"` — could reasonably be `neutral` or `mixed`
- `"I absolutely love getting stuck in traffic 🙃"` — sarcasm makes the true label `negative` despite the word "love"
- `"it is what it is I guess"` — resignation could be read as `neutral` or subtly `negative`

**Important characteristics of your dataset:**
- Contains slang: "lowkey", "highkey", "no cap", "rn"
- Includes Unicode emojis: 🔥 😂 😞 🥲 💀 🙃
- Contains one clear sarcasm example
- Several posts express mixed or ambiguous feelings
- All posts are short (under 15 words), social-media style

**Possible issues with the dataset:**
- Only 14 examples — far too small for reliable generalization
- Label distribution is uneven: more `negative` examples than `neutral` or `mixed`
- No posts longer than one or two sentences
- Slang and emoji vocabulary is limited to a narrow cultural context
- A second human labeler might disagree on 2–3 of the `mixed` and `neutral` labels

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**
- Each token matching a word in `POSITIVE_WORDS` adds +1 to the score; `NEGATIVE_WORDS` subtracts −1.
- Emoji tokens (🔥, 😂, 😞, 💀, etc.) and text emoticons (:), :() score ±2 — weighted higher because emojis are strong, deliberate sentiment signals.
- Negation words (`not`, `no`, `never`, `don't`, and common contractions) set a flag that flips the sign of the immediately following word's score.
- If the token list contains at least one positive signal **and** at least one negative signal (before negation is applied), the label is `mixed` regardless of the total score.
- Otherwise: score > 0 → `positive`, score < 0 → `negative`, score == 0 → `neutral`.
- Preprocessing normalizes repeated characters (`sooooo` → `soo`) and pads emoticons so they tokenize correctly.

**Strengths of this approach:**
- Fully transparent — every prediction can be traced to specific word hits.
- Negation handling correctly flips "not happy" to negative.
- Emoji support gives strong, reliable signals for posts heavy on visual sentiment.
- Easy to improve incrementally by adding words to the vocabulary lists.

**Weaknesses of this approach:**
- Cannot detect sarcasm (`"I absolutely love getting stuck in traffic 🙃"` scores as mixed, not negative).
- Vocabulary-dependent: words not in the lists are ignored entirely (e.g. "hopeful" and "proud" had to be added manually).
- Slang that doesn't map to known words is invisible to the model.
- Negation only covers one word ahead — "not at all happy" would not be handled correctly.

## 4. How the ML Model Works (if used)

**Features used:**
Bag of words using `CountVectorizer`. Each post is represented as a vector of word counts across the full vocabulary of the training set.

**Training data:**
The model trained on all 14 examples in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**
The model achieved 100% accuracy on the training set. This is expected — with only 14 examples and one model parameter per word, logistic regression can perfectly memorize the training data. Accuracy on unseen posts would likely be much lower.

**Strengths and weaknesses:**
Strengths: learns co-occurrence patterns automatically without hand-written rules; correctly classified the sarcasm example because the emoji 🙃 appeared in the training data with a `negative` label.
Weaknesses: with 14 training examples the model is heavily overfitting; it will fail on any sentence using words not seen during training; no understanding of word order, negation, or context.

## 4b. Rule Based vs. ML Model Comparison

**Did the learned model behave differently from the rule based one?**
Yes, in one important case. The ML model correctly predicted `negative` for the sarcasm post `"I absolutely love getting stuck in traffic 🙃"`, while the rule based model predicted `mixed`. The ML model learned that 🙃 co-occurs with `negative` labels in the training data and let that override "love", whereas the rule based model treated both signals as equal and called it mixed. For all other 13 posts, both models agreed.

**Did it fix certain failures or introduce new ones?**
The ML model fixed the one rule based failure (sarcasm post), achieving 14/14 vs. 13/14. However, this is misleading — the ML model is simply memorizing the training data. It did not generalize; it just recalled the exact example it was trained on. If tested on a new sarcastic sentence it had never seen, it would almost certainly fail just as the rule based model does.

**How sensitive was it to the labels you created?**
Very sensitive. With only 14 examples, every label has outsized influence. For instance:
- The `mixed` label appears on 3 posts. If any of those were relabeled `positive` or `negative`, the model's entire concept of "mixed" would shift.
- The model has no fallback — if a new post contains no words from the training vocabulary, logistic regression will predict whichever class had the most training examples (likely `negative`).
- Because training and test data are the same 14 posts, any labeling mistake is both learned and "evaluated" without ever being caught. A bad label looks like a correct prediction.

In short: the ML model is more brittle and less interpretable than the rule based one at this data size, even though its reported accuracy is higher.

## 5. Evaluation

**How you evaluated the model:**
Both models were evaluated on the same 14 labeled posts in `dataset.py` (training accuracy, since no separate test set exists).

**Rule based model final accuracy: 13/14 (93%)**
**ML model accuracy on training data: 14/14 (100%)**

**Examples of correct predictions:**
- `"I am not happy about this"` → `negative`: negation handling correctly flipped "happy" to a negative signal.
- `"Feeling tired but kind of hopeful"` → `mixed`: "tired" (negative) and "hopeful" (positive) both fired, triggering the mixed-detection logic.
- `"No cap this is the best day ever 😂"` → `positive`: emoji 😂 scored +2 and drove the label even though "no cap" is unrecognized slang.

**Examples of incorrect predictions (rule based):**
- `"I absolutely love getting stuck in traffic 🙃"` → predicted `mixed`, true `negative`. See §4b.

## 6. Limitations

- **Dataset too small:** 14 examples cannot represent the diversity of real language. The ML model's 100% training accuracy is a sign of memorization, not learning.
- **No test set:** both models are evaluated on their own training data, so reported accuracy is optimistic.
- **Sarcasm and vocabulary gaps:** see the rule based weaknesses in §3.
- **Short text only:** both models were designed and tested on posts under 15 words. Longer, more complex sentences are untested.
- **Cultural and linguistic bias:** see §6b.

## 6b. Bias and Scope

**What language is this model optimized for?**
The dataset was written entirely in informal American English by one person. The slang used — "lowkey", "highkey", "no cap", "rn" — reflects a specific online, youth-oriented register. The emojis chosen (🔥 😂 💀 🥲 🙃) are common in that same context. The model performs well for someone who writes the way the training data was written.

**Who might it misinterpret?**
- **Speakers of other dialects or Englishes:** AAVE, British slang, Indian English, or code-switching text use different expressions for the same emotions. Words or phrases the model has never seen score zero — the post gets pushed toward `neutral` regardless of actual sentiment.
- **Older or more formal writers:** someone writing "I find this deeply troubling" or "I am quite pleased" uses no words from the word lists and gets labeled `neutral`, which is wrong.
- **Non-native English speakers:** grammatical patterns that differ from standard English may break negation handling (e.g. "I am not at all happy" vs. "not happy I am").
- **Sarcasm-heavy communities:** any community where irony is the default tone (extremely common online) will be systematically misread as positive or mixed.
- **People expressing distress indirectly:** someone writing "I just don't care anymore 💀" might get `negative` by luck (💀 is in the emoji list), but indirect or culturally specific expressions of sadness would be missed entirely.

**Bottom line:** this model was built by and for one narrow slice of informal English. It should not be applied to any population broader than that without retraining on representative data.

## 7. Ethical Considerations

- **Misclassifying distress:** a message expressing genuine emotional pain that uses unusual phrasing could be labeled `neutral` or even `positive`, causing a system built on this model to miss someone who needs support.
- **Language community bias:** slang, dialects, and non-standard English that aren't in the vocabulary are silently ignored. Users whose natural writing style differs from the training data will receive worse predictions.
- **Sarcasm and irony:** misreading a sarcastic negative post as positive (or mixed) could have real consequences in moderation, mental health, or customer-service applications.
- **Privacy:** any system that analyzes personal messages for mood is processing sensitive data. Users should be informed and consent should be explicit.
- **Over-reliance:** a 93% rule-based model on 14 hand-picked examples should not be used to make real decisions about people. Accuracy on unseen, real-world text is unknown.

## 8. Ideas for Improvement

- **Add more labeled data:** at minimum 100–200 diverse examples to reduce overfitting and improve generalization.
- **Add a real test set:** hold out 20% of data before training so reported accuracy reflects performance on unseen examples.
- **Use TF-IDF instead of CountVectorizer:** down-weights common words and up-weights distinctive ones, which often improves text classification.
- **Improve slang and emoji coverage:** expand `POSITIVE_WORDS` / `NEGATIVE_WORDS` with common slang terms, or map slang to sentiment scores explicitly.
- **Multi-word negation:** extend negation handling to cover phrases like "not at all" or "far from happy."
- **Use a small pretrained model:** a fine-tuned sentence transformer (e.g. `sentence-transformers`) would handle sarcasm, context, and unseen vocabulary far better than either current approach.
- **Sarcasm detection heuristic:** flag posts where a strong positive word appears alongside a negative-sentiment emoji (e.g. 🙃, 💀) as candidates for manual review rather than auto-labeling.
