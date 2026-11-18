"""Streaming descriptive statistics without collecting all comments in memory."""
import numpy as np
from src.config import LABELS, TARGETS


class Summary:
    """Accumulate exact totals, histograms and label co-occurrence."""
    def __init__(self) -> None:
        self.rows = self.scored_rows = self.words = self.characters = 0
        self.duplicate_comments = 0
        self.blank_comments = 0
        self.seen_text: set[str] = set()
        self.prevalence = {label: 0 for label in TARGETS}
        self.cooccurrence = np.zeros((6, 6), dtype='int64')
        self.bins = np.array([0, 100, 300, 1000, 3000, np.inf])
        self.histogram = np.zeros(5, dtype='int64')

    def update(self, chunk) -> None:
        self.rows += len(chunk)
        self.words += int(chunk.word_count.sum())
        self.characters += int(chunk.character_count.sum())
        self.blank_comments += int(chunk.comment_text.str.strip().eq('').sum())
        self.histogram += np.histogram(chunk.character_count, bins=self.bins)[0]
        for key in chunk.text_group:
            self.duplicate_comments += int(key in self.seen_text)
            self.seen_text.add(key)
        scored = chunk.loc[(chunk[LABELS] >= 0).all(axis=1)]
        self.scored_rows += len(scored)
        for label in TARGETS:
            self.prevalence[label] += int(scored[label].sum())
        values = scored[LABELS].to_numpy(dtype='int64')
        self.cooccurrence += values.T @ values

    def result(self) -> dict:
        return dict(rows=self.rows, scored_rows=self.scored_rows,
                    unscored_rows=self.rows - self.scored_rows,
                    duplicate_comments=self.duplicate_comments,
                    blank_comments=self.blank_comments,
                    null_comments=0, null_ids=0, duplicate_ids=0,
                    mean_words=self.words / max(self.rows, 1),
                    mean_characters=self.characters / max(self.rows, 1),
                    positives=self.prevalence,
                    prevalence={k: v / max(self.scored_rows, 1) for k, v in self.prevalence.items()},
                    length_bins=['0-99', '100-299', '300-999', '1000-2999', '3000+'],
                    length_counts=self.histogram.tolist(),
                    cooccurrence=self.cooccurrence.tolist())
