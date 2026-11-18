-- name: split_counts
SELECT dataset_split, COUNT(*) AS comments FROM comments GROUP BY dataset_split;

-- name: binary_distribution
SELECT c.dataset_split, l.label_value AS any_toxic, COUNT(*) AS comments
FROM comments c JOIN labels l ON c.comment_id=l.comment_id
WHERE l.label_name='any_toxic' GROUP BY c.dataset_split,l.label_value;

-- name: category_prevalence
SELECT c.dataset_split,l.label_name,COUNT(*) AS scored_comments,
 SUM(l.label_value) AS positives,ROUND(AVG(l.label_value),6) AS prevalence
FROM comments c JOIN labels l ON c.comment_id=l.comment_id
WHERE l.label_name<>'any_toxic' GROUP BY c.dataset_split,l.label_name;

-- name: positive_cooccurrence
SELECT c.dataset_split,a.label_name AS label_a,b.label_name AS label_b,COUNT(*) AS both_positive
FROM labels a JOIN labels b ON a.comment_id=b.comment_id AND a.label_name<b.label_name
JOIN comments c ON c.comment_id=a.comment_id
WHERE a.label_value=1 AND b.label_value=1 AND a.label_name<>'any_toxic' AND b.label_name<>'any_toxic'
GROUP BY c.dataset_split,a.label_name,b.label_name HAVING COUNT(*)>=1
ORDER BY both_positive DESC;

-- name: length_groups
WITH groups AS (
 SELECT c.dataset_split,CASE WHEN character_count<100 THEN '01 short (<100)'
 WHEN character_count<500 THEN '02 medium (100-499)' ELSE '03 long (500+)' END AS length_group,
 l.label_value FROM comments c JOIN labels l ON c.comment_id=l.comment_id
 WHERE l.label_name='any_toxic'
)
SELECT dataset_split,length_group,COUNT(*) AS comments,ROUND(AVG(label_value),6) AS toxic_rate
FROM groups GROUP BY dataset_split,length_group;

-- name: label_ranking
WITH totals AS (
 SELECT c.dataset_split,l.label_name,SUM(l.label_value) AS positives
 FROM labels l JOIN comments c ON c.comment_id=l.comment_id WHERE l.label_name<>'any_toxic'
 GROUP BY c.dataset_split,l.label_name
)
SELECT *,DENSE_RANK() OVER(PARTITION BY dataset_split ORDER BY positives DESC) AS prevalence_rank
FROM totals ORDER BY dataset_split,prevalence_rank,label_name;

-- name: prediction_distribution
SELECT r.run_id,r.model_name,c.dataset_split,p.label_name,COUNT(*) AS predictions,
 ROUND(AVG(p.probability),6) AS mean_probability,SUM(p.predicted_label) AS predicted_positives
FROM predictions p JOIN model_runs r ON r.run_id=p.run_id
JOIN comments c ON c.comment_id=p.comment_id
GROUP BY r.run_id,r.model_name,c.dataset_split,p.label_name;

-- name: high_confidence_errors
WITH errors AS (
 SELECT r.run_id,r.model_name,c.comment_id,c.dataset_split,SUBSTR(c.comment_text,1,180) AS comment_excerpt,
 p.label_name,l.label_value,p.probability,
 CASE WHEN p.predicted_label=1 THEN p.probability ELSE 1-p.probability END AS confidence
 FROM predictions p JOIN model_runs r ON r.run_id=p.run_id
 JOIN comments c ON c.comment_id=p.comment_id
 JOIN labels l ON l.comment_id=p.comment_id AND l.label_name=p.label_name
 WHERE p.predicted_label<>l.label_value
), ranked AS (
 SELECT *,ROW_NUMBER() OVER(PARTITION BY run_id,label_name ORDER BY confidence DESC,comment_id) AS error_rank
 FROM errors WHERE confidence>=0.8
)
SELECT * FROM ranked WHERE error_rank<=5 ORDER BY run_id,label_name,error_rank;
