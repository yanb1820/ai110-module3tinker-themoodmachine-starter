# The Mood Machine

The Mood Machine is a simple text classifier that begins with a rule based approach and can optionally be extended with a small machine learning model. It tries to guess whether a short piece of text sounds **positive**, **negative**, **neutral**, or even **mixed** based on patterns in your data.

This lab gives you hands on experience with how basic systems work, where they break, and how different modeling choices affect fairness and accuracy. You will edit code, add data, run experiments, and write a short model card reflection.

---

## Repo Structure

```plaintext
├── dataset.py         # Starter word lists and example posts (you will expand these)
├── mood_analyzer.py   # Rule based classifier with TODOs to improve
├── main.py            # Runs the rule based model and interactive demo
├── ml_experiments.py  # (New) A tiny ML classifier using scikit-learn
├── model_card.md      # Template to fill out after experimenting
└── requirements.txt   # Dependencies for optional ML exploration
```

---

## Getting Started

1. Open this folder in VS Code.
2. Make sure your Python environment is active.
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the rule-based starter:

    ```bash
    python main.py
    ```

If pieces of the analyzer are not implemented yet, you will see helpful errors that guide you to the TODOs.

To try the ML model later, run:

```bash
python ml_experiments.py
```

---

## What You Will Do

During this lab you will:

- Implement the missing parts of the rule based `MoodAnalyzer`.
- Add new positive and negative words.
- Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- Observe unusual or incorrect predictions and think about why they happen.
- Train a tiny machine learning model and compare its behavior to your rule based system.
- Complete the model card with your findings about data, behavior, limitations, and improvements.
- The goal is to help you reason about how models behave, how data shapes them, and why even small design choices matter.

---

## Tips

- Start with preprocessing before updating scoring rules.
- When debugging, print tokens, scores, or intermediate choices.
- Ask an AI assistant to help create edge case posts or unusual wording.
- Try examples that mislead or confuse your model. Failure cases teach you the most.

---

## Summary

The core concept students need to understand is that every label, word list, and scoring rule is a design choice, and the model reflects exactly what you put into it. Students are most likely to struggle at the neutral versus mixed boundary, where it is easy to mislabel an ambiguous post without realizing the mistake silently shapes every prediction downstream. AI was helpful for generating diverse edge cases such as slang, emojis, and sarcasm that stretch the dataset in useful ways. AI was misleading on the ML model's accuracy, reporting roughly 100 percent without flagging that training and testing on the same 14 posts makes that number meaningless, which is exactly the misconception students need to confront. The most instructive failure is sarcasm, because a post like "I absolutely love getting stuck in traffic" exposes the hard ceiling of word-list and bag-of-words approaches in a way students can see and reason about directly. To guide a stuck student without giving the answer, ask them to call explain() on the failing post and read the token list aloud, since the mismatch between the words the model counted and what the sentence actually means usually becomes obvious before they finish the sentence.
