"""Versioned contextual-study reports; original research artifacts stay intact."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from html import escape
from src.config import ROOT, LABELS, TARGETS
from src.deliverables import markdown_table
from src.utils import read_json, save_json


def _table(frame: pd.DataFrame, styles, width=499):
    data=[list(frame.columns)]+[[f'{v:.4f}' if isinstance(v,float) else str(v) for v in row]
                              for row in frame.itertuples(index=False,name=None)]
    style=styles['BodyText'].clone('cell'); style.fontSize=7.5; style.leading=10
    rows=[[Paragraph(escape(str(v)),style) for v in row] for row in data]
    table=Table(rows,colWidths=[width/len(frame.columns)]*len(frame.columns),repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
       ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#cbd5e1')),('VALIGN',(0,0),(-1,-1),'TOP'),
       ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return KeepTogether([table,Spacer(1,10)])


def generate_contextual_deliverables(root: Path) -> dict:
    latest=read_json(root/'results/contextual/latest.json')
    study_path=root/latest['study']; folder=study_path.parent
    study=read_json(study_path); selected=study['selected_model']; seed=study['protocol']['seed']
    figure_dir=folder/'figures'; figure_dir.mkdir(exist_ok=True)
    comparisons=pd.read_csv(folder/'assessment_comparison.csv')
    per_label=pd.read_csv(folder/'per_label_assessment.csv')
    slices=pd.read_csv(folder/'contextual_slices.csv')
    threshold_csv=root/latest['model_dir']/f'{selected}_threshold_tradeoffs.csv'
    tradeoffs=pd.read_csv(threshold_csv)
    reference=study['assessment']['tfidf_lr_balanced']['default_0.5']
    candidate=study['assessment'][selected]['calibration_thresholds']
    a,b=reference['per_label']['any_toxic'],candidate['per_label']['any_toxic']
    sns.set_theme(style='whitegrid',palette='colorblind')
    figures=[]
    def save(name):
        plt.tight_layout(); plt.savefig(figure_dir/name,dpi=160,bbox_inches='tight'); plt.close(); figures.append(name)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    names=list(study['protocol']['candidates']); x=np.arange(len(names)); width=.38
    auc=comparisons.loc[comparisons.policy.eq('default_0.5')].set_index('model').reindex(names)
    axes[0].bar(x,auc.mean_auc_six); axes[0].set_title('Assessment mean ROC-AUC (six labels)')
    for axis,key,title in [(axes[1],'false_positive_rate','Assessment binary false-positive rate'),
                            (axes[2],'binary_recall','Assessment binary recall')]:
        for i,policy in enumerate(['default_0.5','calibration_thresholds']):
            values=comparisons.loc[comparisons.policy.eq(policy)].set_index('model').reindex(names)
            axis.bar(x+(i-.5)*width,values[key],width,label='0.5' if i==0 else 'Calibration thresholds')
        axis.set_title(title); axis.legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x,names,rotation=35,ha='right')
    save('assessment_comparison.png')
    pivot=per_label.assign(name=per_label.model+' / '+per_label.policy).pivot(index='name',columns='label',values='f1')
    plt.figure(figsize=(11,7)); sns.heatmap(pivot[TARGETS],annot=True,fmt='.3f',vmin=0,vmax=1,cmap='Blues')
    plt.title('Assessment F1 by target and decision policy'); save('per_label_f1.png')
    any_tradeoff=tradeoffs.loc[tradeoffs.label.eq('any_toxic')].sort_values('threshold')
    plt.figure(figsize=(8,4))
    plt.plot(any_tradeoff.threshold,any_tradeoff.precision,label='Calibration precision')
    plt.plot(any_tradeoff.threshold,any_tradeoff.recall,label='Calibration recall')
    plt.axvline(study['frozen_selection']['thresholds']['any_toxic'],linestyle='--',color='black',label='Frozen threshold')
    plt.axhline(.8,linestyle=':',label='Calibration recall floor 0.80')
    plt.xlabel('any_toxic decision threshold'); plt.ylabel('Metric'); plt.ylim(0,1.02); plt.legend()
    plt.title(f'{selected}: threshold selection uses calibration only'); save('threshold_tradeoffs.png')
    chosen_cohorts=['profanity_and_positive_word_cues','criticism_without_profanity_cues','identity_mention_cue','threat_cue']
    rows=[]
    for name,policy,display in [('tfidf_lr_balanced','default_0.5','Word balanced, 0.5'),
                              (selected,'calibration_thresholds','Selected, saved thresholds')]:
        for cohort in chosen_cohorts:
            row=slices.loc[(slices.model==name)&(slices.policy==policy)&(slices.cohort==cohort)].iloc[0]
            rows.append(dict(model=display,cohort=cohort,negative_rows=int(row.human_labeled_negatives),
                             false_positives=int(row.false_positives),false_positive_rate=row.false_positive_rate,
                             positive_rows=int(row.human_labeled_positives),recall=row.recall))
    cohort_table=pd.DataFrame(rows)
    def cohort_values(cohort,model):
        return cohort_table.loc[(cohort_table.cohort==cohort)&(cohort_table.model==model)].iloc[0]
    crit_ref=cohort_values('criticism_without_profanity_cues','Word balanced, 0.5')
    crit_new=cohort_values('criticism_without_profanity_cues','Selected, saved thresholds')
    prof_ref=cohort_values('profanity_and_positive_word_cues','Word balanced, 0.5')
    prof_new=cohort_values('profanity_and_positive_word_cues','Selected, saved thresholds')
    fig,axis=plt.subplots(figsize=(10,4))
    sns.barplot(data=cohort_table,x='cohort',y='false_positive_rate',hue='model',ax=axis)
    axis.set_xticks(np.arange(4),['Profanity + positive words','Criticism cues, no profanity','Identity mentions','Threat cues'],rotation=15,ha='right')
    axis.set_title('Lexical retrieval slices, Jigsaw ground truth (overlapping cohorts)')
    axis.set_ylabel('False-positive rate among annotated negatives'); save('contextual_slices.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    resource=comparisons.loc[comparisons.policy.eq('default_0.5')]
    for axis,key,title in zip(axes,['training_seconds','inference_ms_per_1000','peak_rss_mb'],
                             ['Fit seconds','Inference milliseconds / 1,000 rows','Sampled absolute RSS MiB']):
        axis.bar(resource.model,resource[key]); axis.set_title(title); axis.tick_params(axis='x',rotation=40)
    save('resources.png')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for axis,entry,title in [(axes[0],a,'Balanced word LR, 0.5'),(axes[1],b,'Selected candidate, calibration thresholds')]:
        sns.heatmap(entry['confusion_matrix'],annot=True,fmt='d',cbar=False,ax=axis,cmap='Blues')
        axis.set(title=title,xlabel='Predicted 0/1',ylabel='Actual 0/1')
    save('binary_confusions.png')
    weights=pd.read_csv(root/'results/contextual/original_class_weights.csv')
    old_cases=pd.read_csv(root/'results/contextual/reported_cases_original_models.csv')
    new_cases=pd.read_csv(folder/'reported_cases_after_freeze.csv')
    cases=[]
    for case in [1,2,3]:
        old=old_cases.loc[(old_cases.model=='tfidf_lr_balanced')&(old_cases.case==case)].iloc[0]
        new=new_cases.loc[(new_cases.model==selected)&(new_cases.case==case)].iloc[0]
        cases.append(dict(case=case,old_any_score=old.any_toxic,new_any_score=new.any_toxic_probability,
                          new_threshold=new.any_toxic_threshold,new_any_flag=new.any_toxic_flag,
                          new_hate_score=new.identity_hate_probability,new_hate_flag=new.identity_hate_flag))
    case_table=pd.DataFrame(cases)
    selected_table=comparisons.loc[comparisons.policy.eq('calibration_thresholds'),
            ['model','mean_auc_six','binary_precision','binary_recall','binary_f1','false_positive_rate']]
    threshold_table=pd.DataFrame([{'target':name,'saved_threshold':value} for name,value in study['frozen_selection']['thresholds'].items()])
    paired=study['paired_false_positives']; ci=paired['paired_bootstrap_95_interval']
    narrative=[
      ('1. Scope, preservation and decision',
       f"Study {latest['run_id']}. The original experiment 80bf260546a4, all original models, metrics, figures, reports and SQLite records remain unchanged. "
       f"The calibration-frozen candidate was {selected}. All prespecified assessment guards passed: {study['accepted']}. "
       'This supports a conditional operating-point improvement on Jigsaw annotations, not a claim of reliable contextual or moral understanding. The original dashboard default remains unchanged.'),
      ('2. What caused the reported failures',
       'Direct estimator calls, single-comment inference and batch inference agreed exactly for all four original models. Label order, vectorizer reuse and preprocessing were correct. '
       'The reported scores were reproduced. No prediction-pipeline corruption caused them. Saved per-label thresholds were previously ignored by the API, but every original saved value was 0.5, so that missing feature did not cause these three failures. '
       'Probability estimates from class-weighted LR are uncalibrated. A high score is not a calibrated probability of harmful intent.'),
      ('3. Label ontology and context',
       'The six original labels are toxic, severe_toxic, obscene, threat, insult and identity_hate. any_toxic is a separately trained seventh estimator whose training target is the maximum of those six ground-truth labels. '
       'It includes obscene. Positive speech with profanity can therefore be flagged under this dataset target even if its intent is benign. Negative criticism is not automatically an insult, and identity-related disagreement is not automatically identity hate. '
       'Threats express harm; identity hate targets people because of identity. The current lexical models cannot reliably infer these semantic distinctions. A separate independently adjudicated intent benchmark is required before claiming that ability.'),
      ('4. Independently labeled assessment examples',
       f"The original training source was split into {study['counts']['fit']:,} fit, {study['counts']['calibration']:,} calibration and {study['counts']['assessment']:,} assessment rows, with seed {seed}. "
       'All candidates were refitted on exactly the fit partition. Exact trimmed-text groups stay together. Vectorizers fit on fit text only. '
       'Ground truth consists of pre-existing Jigsaw annotations supplied independently of these model predictions. Assessment rows were saved before scoring; no fresh independent human intent annotation was collected. '
       'The source corpus had been explored in earlier work, so this is a held-out internal assessment rather than a pristine external test. No official-test file or labels were loaded. The three supplied comments were diagnostic-only and were absent from training source.'),
      ('5. Five candidates and architecture',
       'The four original algorithms were refitted as new artifacts for fair comparison. The fifth uses a shared sparse FeatureUnion of word TF-IDF (50,000 unigrams/bigrams) and character-word TF-IDF (30,000 3-5-grams), with weights 1 and 0.5. '
       'Six independent binary logistic estimators implement the multilabel outputs, and a seventh independently predicts any_toxic. Balanced LR uses C=1 and liblinear. This adds local phrase/spelling evidence, not full compositional context. '
       'DistilBERT was not trained. No Transformer performance is claimed. All experiments run on CPU.'),
      ('6. Calibration protocol and frozen selection',
       'For every label, the highest calibration threshold meeting a recall floor was chosen, minimizing calibration false positives. Floors were 0.80 for each target except threat, which used 0.85. '
       'Candidates within 0.01 mean six-label AUC of the balanced-word baseline were eligible; lowest calibration binary false-positive rate chose the candidate. The choice was written to decision_before_assessment.json before assessment metrics were accessed. '
       'Assessment acceptance guards were declared in advance: at least 20% relative FPR reduction, binary recall at least 0.78, threat recall at least 0.80, identity-hate recall at least 0.75, AUC loss no more than 0.01, and a paired FPR interval wholly below zero. '
       'Thresholds never change probabilities or ROC-AUC. Single-class calibration targets retain 0.5. No threshold was chosen for the three supplied examples.'),
      ('7. Held-out assessment results',
       f"Mean six-label ROC-AUC increased from {reference['mean_roc_auc_six']:.6f} to {candidate['mean_roc_auc_six']:.6f}. "
       f"Binary FPR changed from {a['false_positive_rate']:.2%} to {b['false_positive_rate']:.2%}, a {study['relative_fpr_reduction']:.1%} relative reduction. "
       f"Precision changed from {a['precision']:.2%} to {b['precision']:.2%}; recall changed from {a['recall']:.2%} to {b['recall']:.2%}; F1 changed from {a['f1']:.4f} to {b['f1']:.4f}. "
       f"The paired assessment avoided {paired['avoided_false_positives']} false positives and introduced {paired['introduced_false_positives']}, net {paired['avoided_false_positives']-paired['introduced_false_positives']} fewer. "
       f"The 95% paired bootstrap interval for candidate-minus-reference FPR is [{ci[0]:.5f}, {ci[1]:.5f}] using 10,000 draws. This exploratory comparison is not a multiplicity-adjusted global superiority claim."),
      ('8. Rare-label protection and trade-offs',
       f"Threat recall changed from {reference['per_label']['threat']['recall']:.2%} to {candidate['per_label']['threat']['recall']:.2%}; identity-hate recall from {reference['per_label']['identity_hate']['recall']:.2%} to {candidate['per_label']['identity_hate']['recall']:.2%}. "
       f"Selected recall Wilson 95% intervals: {json.dumps(study['selected_recall_wilson95'])}. Rare-label samples are small, so the uncertainty is material. "
       'Lower protected-category thresholds can increase their false positives and reduce their precision/F1. The separately trained binary decision can disagree with category flags; threat/hate flags must be reviewed independently. '
       'Overall binary recall fell by roughly five percentage points. This is a measured trade-off, not domination at every metric.'),
      ('9. Contextual false-positive patterns',
       'Assessment slices are retrieved using broad lexical cues. Regexes never relabel examples or enter inference. Slice names describe cues, not verified intent. Cohorts overlap. '
       f"The criticism-without-profanity slice has {int(crit_ref.negative_rows):,} Jigsaw-labeled negatives: false positives changed from {int(crit_ref.false_positives)} to {int(crit_new.false_positives)}, while binary recall among its {int(crit_ref.positive_rows)} labeled positives changed from {crit_ref.recall:.4f} to {crit_new.recall:.4f}. "
       f"The profanity-plus-positive-word slice has {int(prof_ref.negative_rows)} negatives: false positives changed from {int(prof_ref.false_positives)} to {int(prof_new.false_positives)}. "
       'These results do not establish correct handling of benign profanity or discriminatory meaning.'),
      ('10. Class imbalance and coefficient evidence',
       'Original balanced positive weights were about 4.92 for any_toxic, 56.44 for identity_hate and 172.04 for threat. These emphasize rare positives and alter the operating point. '
       'In the original positive-profanity example, the token fucking contributed +4.17 to any_toxic log odds, while amazing contributed about -0.31. In the criticism example, worst, the worst, have ever and ever all raised log odds. '
       'The religion token raised identity_hate log odds, but this local association did not capture the full denial-of-rights meaning. These are exact linear feature contributions, not causal explanations of language. Full coefficients and reconstructed scores are saved in the audit artifacts.'),
      ('11. The supplied examples remain unresolved',
       'All three supplied sentences were scored only after model/threshold freezing. The fifth model still assigns a high score to positive profanity and flags ordinary criticism. Its saved policy misses the discriminatory sentence, with identity_hate score about 0.039. '
       'This is a concrete regression on a diagnostic case despite better aggregate assessment. No exceptions or substitutions were introduced. Keep the original default and treat the new model as an opt-in experimental operating point. '
       'Fresh independently human-adjudicated intent examples, including benign profanity and discriminatory implications, are needed for a stronger semantic model and external evaluation.'),
      ('12. Resource usage, UI and reproducibility',
       'Fit/inference time and absolute RSS were measured in fresh CPU processes. Inference timing covers both calibration and assessment batches; milliseconds per 1,000 rows is normalized over that fixed workload. Peak RSS includes loaded data and native allocations; 10 ms sampling can miss short peaks and desktop load affects timings. '
       'Live Detection now supports global, saved per-label and custom per-label thresholds, with side-by-side raw scores and separate decisions. Batch prediction can use saved thresholds. The original default, historical metrics, figures and SQLite records remain intact. '
       'Run python -m scripts.contextual_experiment to create a new versioned study, scripts.contextual_diagnostics to verify provenance/preservation, and scripts.contextual_report to regenerate this supplement. New tests use results/contextual/test_results.xml, preserving the original test record.'),
      ('13. References and provenance',
       'Original data and label source: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview . '
       'Threshold/score distinction and separate tuning data: https://scikit-learn.org/stable/modules/classification_threshold.html . '
       'Logistic regression class weighting: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html . '
       'Protocol, model hashes, split assignments, calibration trade-offs, independently annotated assessment rows, per-label metrics and false-positive examples are stored in this study directory. Original artifacts are checked by SHA-256.'),
    ]
    sections=[(title,body) for title,body in narrative]
    markdown='# Contextual error analysis and operating-point study\n\n'+'\n\n'.join(f'## {title}\n\n{body}' for title,body in sections)
    markdown+='\n\n## Assessment comparison (calibration policies)\n\n'+markdown_table(selected_table)
    markdown+='\n\n## Saved thresholds\n\n'+markdown_table(threshold_table)
    markdown+='\n\n## Supplied-case diagnostics, after freezing\n\n'+markdown_table(case_table)
    markdown+='\n\n## Per-label assessment and all operating points\n\nSee `'+str((folder/'per_label_assessment.csv').relative_to(root).as_posix())+'` and `assessment_comparison.csv`.\n'
    (root/'reports/contextual_error_analysis.md').write_text(markdown,encoding='utf-8')
    (root/'docs/contextual_methodology.md').write_text(markdown,encoding='utf-8')
    styles= getSampleStyleSheet(); styles['BodyText'].fontSize=9.5; styles['BodyText'].leading=14
    for key in ['Heading1','Heading2']: styles[key].keepWithNext=True; styles[key].textColor=colors.HexColor('#17283d')
    document=SimpleDocTemplate(str(root/'reports/contextual_error_analysis.pdf'),pagesize=(595.28,841.89),
           leftMargin=48,rightMargin=48,topMargin=48,bottomMargin=48,title='Contextual error analysis and operating-point study')
    story=[Paragraph('Toxic Comment Detection',styles['Title']),Paragraph('Contextual error analysis and operating-point study',styles['Heading1']),
           Paragraph('Versioned supplement. The original report and experiment remain unchanged.',styles['BodyText']),Spacer(1,12)]
    for title,body in sections:
        story.extend([Paragraph(escape(title),styles['Heading1']),Paragraph(escape(body),styles['BodyText']),Spacer(1,10)])
        if title.startswith('6.'):
            story.append(_table(threshold_table,styles))
        if title.startswith('7.'):
            story.append(_table(selected_table,styles))
        if title.startswith('11.'):
            story.append(_table(case_table,styles))
        if title.startswith('12.'):
            story.append(_table(resource[['model','training_seconds','inference_ms_per_1000','peak_rss_mb']],styles))
        images={'7.':['assessment_comparison.png','binary_confusions.png'],
                '8.':['per_label_f1.png'],'9.':['contextual_slices.png'],'6.':['threshold_tradeoffs.png'],'12.':['resources.png']}
        for prefix,names_ in images.items():
            if not title.startswith(prefix):continue
            from PIL import Image as PILImage
            for name in names_:
                path=figure_dir/name
                with PILImage.open(path) as image: width,height=image.size
                story.extend([Image(str(path),width=document.width,height=document.width*height/width),Spacer(1,10)])
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.drawString(48,28,'Contextual study supplement - original experiment preserved')
        canvas.drawRightString(547,28,str(doc.page))
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    _presentation(root,study,selected_table,threshold_table,case_table,figure_dir)
    save_json(folder/'deliverable_status.json',dict(report='reports/contextual_error_analysis.pdf',
              presentation='presentation/contextual_improvements.pptx',figures=figures,accepted=study['accepted']))
    return {'study':latest['run_id'],'accepted':study['accepted'],'figures':len(figures)}


def _presentation(root,study,comparison,thresholds,cases,figure_dir):
    reference=root/'archive/before_contextual_80bf260546a4/presentation/final_presentation.pptx'
    if not reference.exists():
        reference=root/'presentation/final_presentation.pptx'
    deck=Presentation(reference)
    # Reuse the supplied project theme/dimensions, but author the supplement as
    # a distinct file with explicit study evidence and no historical replacement.
    while len(deck.slides):
        slide_id=deck.slides._sldIdLst[0]; deck.part.drop_rel(slide_id.rId); deck.slides._sldIdLst.remove(slide_id)
    def add(title,paragraphs=None,image=None,table=None):
        slide=deck.slides.add_slide(deck.slide_layouts[6]); slide.background.fill.solid(); slide.background.fill.fore_color.rgb=RGBColor.from_string('F6F8FB')
        def text(x,y,w,h,content,size=24,bold=False,color='17283D'):
            shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=shape.text_frame; tf.word_wrap=True
            for i,line in enumerate(content.split('\n')):
                p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line; p.font.name='Arial'; p.font.size=Pt(size); p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(12)
        text(.65,.4,12,.95,title,33,True)
        if image:
            from PIL import Image as PILImage
            with PILImage.open(figure_dir/image) as im:w,h=im.size
            width=11.6;height=min(4.6,width*h/w);width=height*w/h
            slide.shapes.add_picture(str(figure_dir/image),Inches((13.333-width)/2),Inches(1.5),width=Inches(width),height=Inches(height))
            if paragraphs:text(.8,6.2,11.7,.7,'\n'.join(paragraphs),18,color='087F8C')
        elif table is not None:
            shape=slide.shapes.add_table(len(table)+1,len(table.columns),Inches(.7),Inches(1.55),Inches(11.9),Inches(4.0))
            for j,label in enumerate(table.columns):shape.table.cell(0,j).text=str(label)
            for i,row in enumerate(table.itertuples(index=False,name=None),1):
                for j,value in enumerate(row):shape.table.cell(i,j).text=f'{value:.4f}' if isinstance(value,float) else str(value)
            for row in shape.table.rows:
                for cell in row.cells:
                    for p in cell.text_frame.paragraphs:p.font.name='Arial';p.font.size=Pt(16)
            if paragraphs:text(.8,5.9,11.6,1.0,'\n'.join(paragraphs),20,color='087F8C')
        else:
            y=1.6
            for content in paragraphs or []:
                text(.8,y,11.7,1.0,content,24);y+=1.23
        text(12.1,7.0,.6,.25,f'{len(deck.slides):02d}',11,color='526579')
        slide.notes_slide.notes_text_frame.text=('Source: saved study '+study['protocol']['run_id']+'. '+
           'Jigsaw: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview\n'+
           'Threshold tuning: https://scikit-learn.org/stable/modules/classification_threshold.html\n'+
           'All reported metrics use the internal assessment partition. No official test data was read.')
    add('Contextual errors and decision thresholds',['Toxic Comment Detection: versioned improvement study',
          'Five CPU candidates. Separate fit, calibration and assessment data.',
          'Original models, results, figures and database records preserved.'])
    add('Pipeline check and target meaning',['Direct, single and batch scores agree exactly. No label-order or vectorizer bug was found.',
          'any_toxic includes obscene. A profanity flag does not establish abusive intent.',
          'Saved thresholds were previously unused, but all original values were 0.5. That did not cause the three reported errors.'])
    add('Separated data and frozen choices',[f"Fit {study['counts']['fit']:,}; calibration {study['counts']['calibration']:,}; assessment {study['counts']['assessment']:,}. Seed 20261117.",
          'All five candidates refitted on the same fit rows. Jigsaw labels predate these predictions.',
          'Calibration chose thresholds and candidate before assessment metrics. No official-test labels or supplied comments entered tuning.'])
    add('The fifth model',['Word TF-IDF: 50,000 unigram/bigram features. Character-word TF-IDF: 30,000 features.',
          'Six independent binary logistic outputs, plus a separately trained any_toxic output.',
          'Local phrase and spelling evidence improves coverage. This remains a lexical model. DistilBERT was not trained.'])
    add('Calibration threshold trade-offs',['Thresholds change decisions, never probabilities or ROC-AUC.'],image='threshold_tradeoffs.png')
    display=comparison[['model','mean_auc_six','binary_precision','binary_recall','false_positive_rate']].copy()
    display.columns=['Model','Mean AUC (6)','Binary precision','Binary recall','Binary FPR']
    add('Assessment at calibration policies',['The character candidate was selected on calibration, not this table.'],table=display)
    reference=study['assessment']['tfidf_lr_balanced']['default_0.5']
    chosen=study['assessment'][study['selected_model']]['calibration_thresholds']
    add('Measured false positives and misses',[f"Relative binary FPR reduction: {study['relative_fpr_reduction']:.0%}. Binary recall changes from {reference['per_label']['any_toxic']['recall']:.1%} to {chosen['per_label']['any_toxic']['recall']:.1%}."],image='binary_confusions.png')
    label_summary=pd.DataFrame([{'Target':label,'Reference F1':reference['per_label'][label]['f1'],
                     'Selected F1':chosen['per_label'][label]['f1'],'Selected recall':chosen['per_label'][label]['recall']}
                     for label in LABELS])
    add('Per-label trade-offs and rare positives',[f"Threat recall {chosen['per_label']['threat']['recall']:.1%}, identity-hate recall {chosen['per_label']['identity_hate']['recall']:.1%}. Small positive counts imply substantial uncertainty."],table=label_summary)
    add('Contextual slices remain mixed',['Criticism-cue false positives: 91 to 49. Profanity + positive-word cues: 9 to 9. Cues do not establish intent.'],image='contextual_slices.png')
    c=cases[['case','old_any_score','new_any_score','new_any_flag','new_hate_score']].copy();c.columns=['Supplied case','Old any score','New any score','New any flag','New hate score']
    add('The supplied failures are not solved',['Positive profanity and criticism remain flagged. The discriminatory sentence is missed by the saved policy.'],table=c)
    resources=pd.DataFrame([{'Model':r['model_name'],'Fit seconds':r['training_seconds'],
                  'Inference ms / 1,000':r['inference_ms_per_1000'],'Peak RSS MiB':r['peak_rss_mb']} for r in study['runs']])
    add('Resource costs and side-by-side review',['Inference is slower and peak memory is higher. Saved per-label and custom policies are available in Live Detection.'],table=resources)
    add('Evidence, limits and next steps',['Aggregate Jigsaw assessment improved under declared recall guards. This does not demonstrate general contextual understanding.',
          'Keep the original default. The new candidate is opt-in and still makes serious semantic errors.',
          'Collect independent human intent annotations, evaluate externally, and calibrate operating costs before deployment. Full sources and protocol are in the supplement.'])
    deck.save(root/'presentation/contextual_improvements.pptx')
