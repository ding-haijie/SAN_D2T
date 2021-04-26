from collections import Counter
"""
make spelling correction based on naive bayes' theorem: P(w|c) = P(c|w) * P(w) / P(c).
w: real word, c: wrong word
"""


def edits1(word):
    """all edits that are one-edit-distance away from the word"""
    letters = 'abcdefghijklmnopqrstuvwxyz|/'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """all edits that are two-edits-distance away from the word"""
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


class SpellCorrector(object):
    def __init__(self):
        with open('../data/field_cnt.txt', 'r', encoding='utf-8') as f:
            self.field_counter = f.readlines()

        f_dict = {}
        for line in self.field_counter:
            field, count = line.strip().split('\t')
            f_dict[field] = int(count)

        self.FIELD = Counter(f_dict)
        self.SUM = sum(self.FIELD.values())

    def prob(self, word):
        return self.FIELD[word] / self.SUM

    def known(self, words):
        """the subset of words that appear in the dictionary of FIELD"""
        return set(w for w in words if w in self.FIELD)

    def candidates(self, word):
        """
        Generate possible spelling corrections for word,
        simply make a hypothesis here that direct candidates of the word is
        more probable than candidates of that are edits distance away from the word
        """
        return self.known([word]) or self.known(edits1(word)) or self.known(
            edits2(word)) or [word]

    def correct(self, word):
        """most probable spelling correction for word"""
        return max(self.candidates(word), key=self.prob)


if __name__ == '__main__':
    corrector = SpellCorrector()
    print(corrector.correct('namaeas'))  # names
