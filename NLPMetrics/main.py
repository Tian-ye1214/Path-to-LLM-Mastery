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
import matplotlib.pyplot as plt
import os
from scipy.stats import skew, kurtosis
import seaborn as sns
nltk.download('wordnet')


class TextEvaluator:
    def __init__(self, all_evaluation_indicators=None):
        self.pred_data = None
        self.truth_data = None
        if all_evaluation_indicators is None:
            all_evaluation_indicators = ['METEOR', 'RIBES', 'GLEU', 'NIST', 'chrF', 'chrF++', 'BLEU-1', 'BLEU-2',
                                         'BLEU-3', 'BLEU-4', 'CIDEr', 'ROUGE', 'TER']
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

    def calculate_metrics(self, truth, generated):
        scores = {}
        hypothesis_scores = []
        for gene_sentence in generated:
            res_token = [[res for res in jieba.cut(truth_sentence) if res.strip() and res not in self.punctuation] for truth_sentence in truth]
            hyp_tokens = [h for h in jieba.cut(gene_sentence) if h.strip() and h not in self.punctuation]
            gts = {'caption': [' '.join(res) for res in res_token]}
            res = {'caption': [' '.join(hyp_tokens)]}

            metric_scores = {}
            if 'METEOR' in self.all_evaluation_indicators:
                metric_scores['METEOR'] = meteor_score.meteor_score(res_token, hyp_tokens)
            if 'GLEU' in self.all_evaluation_indicators:
                metric_scores['GLEU'] = gleu_score.sentence_gleu(res_token, hyp_tokens)
            if 'NIST' in self.all_evaluation_indicators:
                metric_scores['NIST'] = nist_score.sentence_nist(res_token, hyp_tokens)
            if 'RIBES' in self.all_evaluation_indicators:
                metric_scores['RIBES'] = ribes_score.sentence_ribes([truth_sentence for truth_sentence in truth], gene_sentence)
            if 'chrF' in self.all_evaluation_indicators:
                metric_scores['chrF'] = sacrebleu.sentence_chrf(hypothesis=gene_sentence, references=[truth_sentence for truth_sentence in truth]).score / 100.0
            if 'chrF++' in self.all_evaluation_indicators:
                metric_scores['chrF++'] = sacrebleu.sentence_chrf(hypothesis=gene_sentence, references=[truth_sentence for truth_sentence in truth], word_order=2).score / 100.0
            if 'TER' in self.all_evaluation_indicators:
                metric_scores['TER'] = sacrebleu.sentence_ter(hypothesis=gene_sentence, references=[truth_sentence for truth_sentence in truth]).score / 100.0

            for scorer_name, scorer in self.scorers.items():
                score_value, _ = scorer.compute_score(gts, res)
                if scorer_name == 'BLEU':
                    for j, bleu_score in enumerate(score_value, start=1):
                        metric_scores[f'BLEU-{j}'] = bleu_score
                else:
                    metric_scores[scorer_name] = score_value

            hypothesis_scores.append(metric_scores)

        for metric in self.all_evaluation_indicators:
            scores[metric] = np.max([h_score.get(metric, 0) for h_score in hypothesis_scores])
        return scores

    def plot_metrics(self, sentence_scores):
        output_dir = 'plots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for metric_name, scores in sentence_scores.items():
            if not scores:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(ax=ax, x=[metric_name]*len(scores), y=scores, hue=[metric_name]*len(scores), palette="muted", legend=False)
            ax.set_title(f'{metric_name} Violin Plot')
            ax.set_ylabel('Score')
            ax.set_xlabel('')

            stats_text = (
                f"Maximum: {np.max(scores):.4f}\n"
                f"Minimum: {np.min(scores):.4f}\n"
                f"Average: {np.mean(scores):.4f}\n"
                f"Variance: {np.var(scores):.4f}\n"
                f"Median: {np.median(scores):.4f}\n"
                f"Std Dev: {np.std(scores):.4f}\n"
                f"Skewness: {skew(scores):.4f}\n"
                f"Kurtosis: {kurtosis(scores):.4f}"
            )

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

            safe_metric_name = "".join(c for c in metric_name if c.isalnum() or c in ('-', '_')).rstrip()
            plot_path = os.path.join(output_dir, f'{safe_metric_name}_violinplot.png')
            plt.savefig(plot_path)
            plt.close(fig)

    def evaluate(self, truth_path, pred_path):
        self.load_data(truth_path, pred_path)
        sentence_scores = {metric: [] for metric in self.all_evaluation_indicators}

        for key, truths in tqdm(self.truth_data.items(), desc='calculating metrics'):
            if key in self.pred_data:
                generated = self.pred_data[key]
                scores = self.calculate_metrics(truths, generated)
                for metric, score in scores.items():
                    sentence_scores[metric].append(score)

        self.metrics.update({
            metric: np.mean(scores)
            for metric, scores in sentence_scores.items()
        })
        self.plot_metrics(sentence_scores)

        return self.metrics

    def print_results(self):
        print("\n=== Evaluation ===")
        for metric in sorted(self.metrics.keys()):
            print(f"{metric}: {self.metrics[metric]:.4f}")


if __name__ == "__main__":
    evaluator = TextEvaluator()
    metrics = evaluator.evaluate('examples/original_captions.json', 'examples/generated_captions.json')
    evaluator.print_results()
