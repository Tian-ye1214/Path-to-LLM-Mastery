import json
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from nltk.translate import meteor_score, nist_score, ribes_score, gleu_score
import sacrebleu
from tqdm import tqdm
import nltk
import jieba
import string
nltk.download('wordnet')


class TextEvaluator:
    def __init__(self, all_evaluation_indicators=None):
        self.pred_data = None
        self.truth_data = None
        if all_evaluation_indicators is None:
            all_evaluation_indicators = ['METEOR', 'RIBES', 'GLEU', 'NIST', 'chrF', 'chrF++', 'BLEU-1', 'BLEU-2', 
                                         'BLEU-3', 'BLEU-4', 'CIDEr', 'ROUGE']
        self.all_evaluation_indicators = all_evaluation_indicators
        self.metrics = {}
        self.scorers = {
            'BLEU': Bleu(n=4),
            'CIDEr': Cider(),
            'ROUGE': Rouge()
        }
        self.punctuation = string.punctuation + "，。！？【】（）《》“”‘’；：——@#￥%……&*{}、|/·~"

    def load_data(self, truth_path, pred_path):
        with open(truth_path, 'r', encoding='utf-8') as f:
            self.truth_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            self.pred_data = json.load(f)

    def calculate_metrics(self, references, hypotheses):
        scores = {}
        hypothesis_scores = []
        for hyp in hypotheses:
            for reference in references:
                res_token = [res for res in jieba.cut(reference) if res.strip() and res not in self.punctuation]
                hyp_tokens = [h for h in jieba.cut(hyp) if h.strip() and h not in self.punctuation]
                metric_scores = {
                    'METEOR': meteor_score.meteor_score([res_token], hyp_tokens),
                    'GLEU': gleu_score.sentence_gleu([res_token], hyp_tokens),
                    'chrF': sacrebleu.sentence_chrf(hypothesis=hyp, references=[reference]).score / 100.0,
                    'chrF++': sacrebleu.sentence_chrf(hypothesis=hyp, references=[reference], word_order=2).score / 100.0,
                    'NIST': nist_score.sentence_nist([res_token], hyp_tokens),
                    'RIBES': ribes_score.sentence_ribes([res_token], hyp_tokens)
                }

                for scorer_name, scorer in self.scorers.items():
                    gts = {'caption': [' '.join(jieba.cut(reference))]}
                    res = {'caption': [' '.join(jieba.cut(hyp))]}
                    score_value, _ = scorer.compute_score(gts, res)

                    if scorer_name == 'BLEU':
                        for j, bleu_score in enumerate(score_value, start=1):
                            metric_scores[f'BLEU-{j}'] = bleu_score
                    else:
                        metric_scores[scorer_name] = score_value

                hypothesis_scores.append(metric_scores)

        for metric in self.all_evaluation_indicators:
            scores[metric] = np.max([h_score[metric] for h_score in hypothesis_scores])
        return scores

    def evaluate(self, truth_path, pred_path):
        self.load_data(truth_path, pred_path)
        sentence_scores = {metric: [] for metric in self.all_evaluation_indicators}

        for key, refs in tqdm(self.truth_data.items(), desc='calculating metrics'):
            if key in self.pred_data:
                hyps = self.pred_data[key]
                scores = self.calculate_metrics(refs, hyps)
                for metric, score in scores.items():
                    sentence_scores[metric].append(score)

        self.metrics.update({
            metric: np.mean(scores)
            for metric, scores in sentence_scores.items()
        })

        return self.metrics

    def print_results(self):
        print("\n=== Evaluation ===")
        for metric in sorted(self.metrics.keys()):
            print(f"{metric}: {self.metrics[metric]:.4f}")


if __name__ == "__main__":
    truth_path = 'examples/original_captions.json'
    pred_path = 'examples/generated_captions.json'
    evaluator = TextEvaluator()
    metrics = evaluator.evaluate(truth_path, pred_path)
    evaluator.print_results()
