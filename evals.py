import nltk
from pyrouge import Rouge155


class BleuScore(object):
    def __init__(self):
        self.list_gens = []
        self.list_refs = []

    def reset_gens(self):
        self.list_gens = []

    def set_refs(self, list_refs):
        self.list_refs = list_refs

    def add_gen(self, text_gen):
        self.list_gens.append(text_gen)

    def calculate(self):
        hypotheses = []
        list_of_references = []
        assert (len(self.list_refs) == len(self.list_gens))
        for text_gen, text_ref in zip(self.list_gens, self.list_refs):
            hypotheses.append(
                [token for token in text_gen.split()]
            )
            list_of_references.append(
                [
                    [token for token in text_ref.split()]
                ]
            )
        # nltk default cumulative weights is (0.25, 0.25, 0.25, 0.25)
        bleu_score = nltk.translate.bleu_score.corpus_bleu(
            list_of_references=list_of_references, hypotheses=hypotheses)
        return bleu_score * 100


class RougeScore(object):
    def __init__(self, system_dir, model_dir, n_gram=4):
        self.system_dir = system_dir  # system path
        self.model_dir = model_dir  # gold path
        self.n_gram = n_gram

        self.list_refs = []
        self.list_gens = []

    def reset_gens(self):
        self.list_gens = []

    def set_refs(self, list_refs):
        self.list_refs = list_refs

    def add_gen(self, text_gen):
        self.list_gens.append(text_gen)

    def file_writer(self):
        for idx in range(len(self.list_gens)):
            with open(self.system_dir + '/text.' + str(idx) + '.txt', 'w') as f:
                f.write(self.list_gens[idx])
        for idx in range(len(self.list_refs)):
            with open(self.model_dir + '/text.A.' + str(idx) + '.txt', 'w') as f:
                f.write(self.list_refs[idx])

    def calculate(self):
        r = Rouge155()
        r.system_dir = self.system_dir
        r.model_dir = self.model_dir
        r.system_filename_pattern = r'text.(\d+).txt'
        r.model_filename_pattern = r'text.A.#ID#.txt'

        output = r.convert_and_evaluate()
        output_dict = r.output_to_dict(output)

        """ look for the f-measure scores """
        rouge_f_score = {}
        for ngram in range(self.n_gram):
            rouge_f_score['rouge_' + str(ngram + 1) + '_f_score'] = 100 * output_dict[
                'rouge_' + str(ngram + 1) + '_f_score']
        rouge_f_score['rouge_l_f_score'] = 100 * output_dict['rouge_l_f_score']

        return rouge_f_score
