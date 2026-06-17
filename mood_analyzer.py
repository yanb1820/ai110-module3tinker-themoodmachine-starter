# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that flip the sentiment of the next word.
_NEGATION_WORDS = {"not", "no", "never", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "can't", "won't"}

# Emoji / text-emoticon sentiment signals.
_POSITIVE_EMOJIS = {":)", ":-)", ":d", "😊", "😄", "😂", "❤️", "🔥"}
_NEGATIVE_EMOJIS = {":(", ":-(", "😞", "😢", "😠", "🥲", "💀", "🙃"}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.

    Known limitations:
      - Sarcasm: phrases like "I absolutely love getting stuck in traffic" score
        as positive (or mixed with a negative emoji) because the classifier reads
        words literally. Detecting sarcasm requires contextual understanding that
        a word-list approach cannot provide.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Lowercase and strip surrounding whitespace.
          - Normalize runs of a repeated character: "sooooo" -> "soo".
          - Pad text emoticons (":)", ":(") so they survive as their own tokens.
          - Drop ASCII punctuation while keeping apostrophes (for "don't"),
            colons (for emoticons), and any non-ASCII characters (emojis).
        """
        cleaned = text.strip().lower()
        # Collapse 3+ repeats of a character down to 2: "sooooo" -> "soo".
        cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
        # Pad text emoticons so they become their own tokens after split.
        for emoticon in (":)", ":-(", ":-)", ":("):
            cleaned = cleaned.replace(emoticon, f" {emoticon} ")
        # Keep word chars, whitespace, apostrophe, colon, and non-ASCII (emojis,
        # via \u0080-\U0010FFFF); replace every other character with a space.
        cleaned = re.sub(r"[^\w\s':\u0080-\U0010FFFF]", ' ', cleaned)
        return cleaned.split()

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def _token_weight(self, token: str) -> int:
        """
        Return the raw sentiment weight of a single token (before negation).

        Emojis are stronger signals than plain words:
          positive emoji -> +2, negative emoji -> -2,
          positive word  -> +1, negative word  -> -1,
          anything else  ->  0.
        """
        if token in _POSITIVE_EMOJIS:
            return 2
        if token in _NEGATIVE_EMOJIS:
            return -2
        if token in self.positive_words:
            return 1
        if token in self.negative_words:
            return -1
        return 0

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Single pass over the tokens that every public method builds on.

        Returns a tuple of:
          - score: total mood score, with negation words flipping the sign
            of the token that immediately follows them.
          - positive_hits: tokens that carried positive sentiment (raw, i.e.
            recorded before negation is applied).
          - negative_hits: tokens that carried negative sentiment (raw).
        """
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negate_next = False

        for token in self.preprocess(text):
            if token in _NEGATION_WORDS:
                negate_next = True
                continue

            weight = self._token_weight(token)
            if weight > 0:
                positive_hits.append(token)
            elif weight < 0:
                negative_hits.append(token)

            score += -weight if negate_next else weight
            negate_next = False

        return score, positive_hits, negative_hits

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words/emojis raise the score, negative ones lower it, and a
        negation word ("not", "never", ...) flips the sign of the next token.
        """
        score, _positive_hits, _negative_hits = self._analyze(text)
        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        We add a "mixed" label on top of the sign-based mapping: if the text
        contains at least one positive signal AND one negative signal, it is
        "mixed" regardless of the total score.
        """
        score, positive_hits, negative_hits = self._analyze(text)

        # Both positive and negative signals present -> mixed sentiment.
        if positive_hits and negative_hits:
            return "mixed"

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Uses the same single-pass analysis as score_text/predict_label, so the
        score and word hits shown here always match the actual prediction.

        Example:
          'Score = 2 (positive: ['love', 'great'], negative: [])'
        """
        score, positive_hits, negative_hits = self._analyze(text)
        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
