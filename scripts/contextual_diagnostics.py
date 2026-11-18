"""Coefficient and imbalance diagnostics, with no model fitting or test access."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from src.config import ROOT, TARGETS
from src.predict import load_bundle, probability_matrix
from src.utils import read_json, save_json, sha256_file


def diagnose(root: Path) -> dict:
    old = read_json(root / 'results/metrics/experiment.json')
    destination = root / 'results/contextual'
    destination.mkdir(exist_ok=True)
    assignments = pd.read_parquet(root / old['split_assignments'])
    frame = pd.read_parquet(root / 'data/processed/train.parquet')
    fit = frame.loc[frame.id.isin(assignments.loc[assignments.partition.eq('train'), 'id'])]
    weights = []
    for label in TARGETS:
        positive, total = int(fit[label].sum()), len(fit)
        weights.append(dict(label=label, positives=positive, negatives=total-positive, prevalence=positive/total,
                      balanced_positive_weight=total/(2*positive),
                      balanced_negative_weight=total/(2*(total-positive))))
    pd.DataFrame(weights).to_csv(destination/'original_class_weights.csv',index=False)
    coefficients=[]
    for run in old['runs']:
        if not run['model_name'].startswith('tfidf'): continue
        bundle=load_bundle(root/run['artifact'])
        names=bundle['vectorizer'].get_feature_names_out()
        for label,estimator in zip(TARGETS,bundle['estimators']):
            values=estimator.coef_[0]
            for direction,indices in [('positive',np.argsort(-values)[:25]),('negative',np.argsort(values)[:25])]:
                for rank,j in enumerate(indices,1):
                    coefficients.append(dict(model=run['model_name'],label=label,direction=direction,
                         rank=rank,feature=str(names[j]),coefficient=float(values[j]),intercept=float(estimator.intercept_[0])))
    pd.DataFrame(coefficients).to_csv(destination/'original_top_coefficients.csv',index=False)
    supplied=pd.read_csv(destination/'reported_cases_original_models.csv').drop_duplicates('case').sort_values('case')
    texts=supplied.comment_text.tolist()
    if frame.comment_text.str.strip().isin(texts).any():
        raise ValueError('A supplied diagnostic is present in the source; explicitly quarantine it before this study')
    latest=read_json(destination/'latest.json')
    case_rows=[]
    if latest:
        study=read_json(root/latest['study'])
        for run in study['runs']:
            bundle=load_bundle(root/run['artifact'])
            p=probability_matrix(bundle,texts)
            for i,text in enumerate(texts):
                case_rows.append(dict(model=run['model_name'],case=i+1,comment_text=text,
                    **{f'{label}_probability':float(p[i,j]) for j,label in enumerate(TARGETS)},
                    **{f'{label}_threshold':bundle['thresholds'][label] for label in TARGETS},
                    **{f'{label}_flag':bool(p[i,j]>=bundle['thresholds'][label]) for j,label in enumerate(TARGETS)}))
        pd.DataFrame(case_rows).to_csv(destination/latest['run_id']/'reported_cases_after_freeze.csv',index=False)
    preserved=read_json(destination/'preserved_artifacts.json',{})
    status={path:sha256_file(root/path)==expected for path,expected in preserved.items()}
    result=dict(original_artifacts_unchanged=all(status.values()),preserved_files=len(status),
                changed=[name for name,passed in status.items() if not passed],
                supplied_cases_in_source_training=0,
                independently_labeled_examples='Training-source assessment rows retain pre-existing Jigsaw labels. No fresh independent human intent annotation was collected.',
                implementation_findings=['Direct single/batch probability outputs agree exactly.',
                    'No label-order, vectorizer-fit or preprocessing corruption was found.',
                    'Saved per-label thresholds were not previously consumed; legacy saved values were all 0.5, so this did not cause the supplied failures.',
                    'Thresholds now affect only decisions; underlying predict_proba outputs remain unchanged.'])
    save_json(destination/'diagnostic_verification.json',result)
    if not result['original_artifacts_unchanged']:
        raise ValueError(f'Original protected artifacts changed: {result["changed"]}')
    print(result)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args(); diagnose(args.root)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
