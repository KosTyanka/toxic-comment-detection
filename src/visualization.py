"""Reproducible figures generated only from observed statistics and metrics."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.config import LABELS, TARGETS, directories
from src.evaluate import comparison_frame
from src.utils import read_json


def generate_figures(root: Path) -> list[str]:
    paths = directories(root)
    outputs = []
    sns.set_theme(style='whitegrid', palette='colorblind')

    def save(name):
        plt.tight_layout()
        plt.savefig(paths['figures'] / name, dpi=160, bbox_inches='tight')
        plt.close()
        outputs.append(name)

    manifest = read_json(paths['processed'] / 'manifest.json')
    if manifest:
        stats = manifest['splits']['train']
        plt.figure(figsize=(9, 4))
        plt.bar(LABELS, [stats['prevalence'][k] for k in LABELS])
        plt.ylabel('Fraction of training comments')
        plt.xticks(rotation=20)
        plt.title('Observed class imbalance in the supplied training set')
        save('label_prevalence.png')
        plt.figure(figsize=(7, 5))
        sns.heatmap(np.array(stats['cooccurrence']), annot=True, fmt='d', xticklabels=LABELS,
                    yticklabels=LABELS, cmap='Blues')
        plt.title('Positive-label co-occurrence (training set)')
        save('cooccurrence.png')
        plt.figure(figsize=(8, 4))
        plt.bar(stats['length_bins'], stats['length_counts'])
        plt.xlabel('Characters per comment')
        plt.ylabel('Training comments')
        save('comment_lengths.png')
    experiment = read_json(paths['metrics'] / 'experiment.json')
    if experiment:
        table = comparison_frame(experiment)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for axis, column, title in zip(axes, ['mean_ROC_AUC_6', 'training_s', 'peak_RSS_MiB'],
                                      ['Validation mean ROC-AUC (six labels)', 'Training seconds', 'Sampled peak RSS (MiB)']):
            axis.bar(table.model, table[column])
            axis.set_title(title)
            axis.tick_params(axis='x', rotation=35)
        save('model_comparison.png')
        for label in TARGETS:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for run in experiment['runs']:
                entry = run['validation']['per_label'][label]
                if entry['roc']:
                    axes[0].plot(entry['roc']['fpr'], entry['roc']['tpr'], label=run['model_name'])
                if entry['pr']:
                    axes[1].plot(entry['pr']['recall'], entry['pr']['precision'], label=run['model_name'])
            axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title=f'{label}: ROC')
            axes[1].set(xlabel='Recall', ylabel='Precision', title=f'{label}: PR')
            for axis in axes:
                if axis.lines:
                    axis.legend(fontsize=8)
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1.02)
            save(f'curves_{label}.png')
        winner = next(r for r in experiment['runs'] if r['model_name'] == experiment['selected_model'])
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for axis, label in zip(axes.flat, TARGETS):
            sns.heatmap(winner['validation']['per_label'][label]['confusion_matrix'], annot=True, fmt='d',
                        cbar=False, ax=axis, cmap='Blues')
            axis.set(title=label, xlabel='Predicted (0/1)', ylabel='Actual (0/1)')
        axes.flat[-1].axis('off')
        save('confusion_matrices.png')
        official = read_json(paths['metrics'] / 'official_test' / f"{experiment['experiment_id']}.json")
        if official and official.get('status') == 'evaluated':
            fig, axes = plt.subplots(2, 4, figsize=(15, 7))
            for axis, label in zip(axes.flat, TARGETS):
                sns.heatmap(official['evaluation']['per_label'][label]['confusion_matrix'],
                            annot=True, fmt='d', cbar=False, ax=axis, cmap='Blues')
                axis.set(title=label, xlabel='Predicted (0/1)', ylabel='Actual (0/1)')
            axes.flat[-1].axis('off')
            fig.suptitle('Selected model: eligible official test rows', y=1.01)
            save('official_test_confusion_matrices.png')
    return outputs
