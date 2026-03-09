# Model Tuning Report

- Tuning enabled: `True`
- Validation split: `0.1`
- Random state: `42`
- Max round: `2`
- Selection metric: `rmse`
- Selected model: `elasticnet-a01-l03` (`elasticnet`)

## Candidate Results

| rank | round | name | regressor | status | selection_metric | selection_value | mae | rmse | custom_score | alpha | l1_ratio | variance_threshold | n_components | description | usual_use | good_points | bad_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | elasticnet-a01-l03 | elasticnet | ok | rmse | 2.9373043305924282 | 2.367310652644013 | 2.9373043305924282 | 2.5383087560285373 | 0.01 | 0.3 | nan | nan | ElasticNet with lighter regularization. | Signal-rich settings with many useful features. | Less shrinkage can improve fit. | Higher overfitting risk. |
| 2 | 1 | elasticnet-default | elasticnet | ok | rmse | 3.0661168201047397 | 2.436119198705511 | 3.0661168201047397 | 2.6251184851252796 | 0.1 | 0.5 | nan | nan | Linear model with mixed L1/L2 regularization. | High-dimensional data where feature selection helps. | Can shrink coefficients and set some to zero. | Needs alpha/l1_ratio tuning and can be slow. |
| 3 | 1 | ridge-alpha-01 | ridge | ok | rmse | 3.1807090934726374 | 2.483829003945395 | 3.1807090934726374 | 2.6928930308035675 | 0.1 | nan | nan | nan | Ridge model with lighter L2 penalty. | Low-bias regularized linear baseline. | Stable and fast. | No feature pruning. |
| 4 | 1 | ridge-alpha-1 | ridge | ok | rmse | 3.1808542564713376 | 2.4839981149635433 | 3.1808542564713376 | 2.6930549574158813 | 1.0 | nan | nan | nan | Linear model with L2 regularization. | Correlated features and p >> n settings. | Stable, fast, and handles multicollinearity well. | Does not perform feature selection. |
| 5 | 1 | ridge-alpha-10 | ridge | ok | rmse | 3.1823089768975374 | 2.485688303388432 | 3.1823089768975374 | 2.6946745054411636 | 10.0 | nan | nan | nan | Linear model with stronger L2 regularization. | Noisy or very high-dimensional regression. | Stronger shrinkage can improve generalization. | Can underfit if regularization is too strong. |
| 6 | 1 | elasticnet-a03-l07 | elasticnet | ok | rmse | 3.2370386036336236 | 2.5691449600873426 | 3.2370386036336236 | 2.7695130531512264 | 0.03 | 0.7 | nan | nan | ElasticNet with balanced shrinkage and sparsity. | Correlated sparse predictors. | Good bias/variance compromise. | Still needs robust tuning. |
| 7 | 2 | pca200-elasticnet | pca_elasticnet | ok | rmse | 3.310199680145963 | 2.588872673560283 | 3.310199680145963 | 2.8052707755359867 | 0.05 | 0.5 | nan | 200.0 | PCA(200) followed by ElasticNet. | Dimensionality reduction with sparse linear fit. | Combines latent compression and regularization. | More expensive than plain linear models. |
| 8 | 1 | elasticnet-a005-l09 | elasticnet | ok | rmse | 3.3131101851862947 | 2.6284979430744095 | 3.3131101851862947 | 2.833881615707975 | 0.05 | 0.9 | nan | nan | ElasticNet with stronger L1 component. | Sparse high-dimensional feature selection. | Can remove many weak features. | Can be unstable if over-sparse. |
| 9 | 1 | lasso-a01 | lasso | ok | rmse | 3.439006411691294 | 2.8120179844915008 | 3.439006411691294 | 3.0001145126514386 | 0.01 | nan | nan | nan | Pure L1 regularized linear model. | Automatic feature selection. | Very interpretable sparse coefficients. | Can underperform when features are correlated. |
| 10 | 2 | pca100-ridge | pca_ridge | ok | rmse | 3.665958688808652 | 2.828716931587665 | 3.665958688808652 | 3.0798894587539607 | 1.0 | nan | nan | 100.0 | PCA(100) followed by Ridge. | Preserve more variance before regression. | Can retain richer signal. | Higher complexity and overfitting risk. |
| 11 | 2 | pca50-ridge | pca_ridge | ok | rmse | 3.816965269068273 | 3.0387971702951866 | 3.816965269068273 | 3.2722475999271126 | 1.0 | nan | nan | 50.0 | PCA(50) followed by Ridge. | Compact latent representation for p >> n. | Fast and robust. | Unsupervised reduction may lose target signal. |
| 12 | 2 | varthresh-elasticnet | varthresh_elasticnet | ok | rmse | 8.035513165013414 | 6.567274894557785 | 8.035513165013414 | 7.007746375694474 | 0.05 | 0.5 | 0.01 | nan | Variance filtering then ElasticNet. | Remove low-information CpGs before fitting. | Can reduce noise and dimensionality. | Can discard weak but useful features. |

## Cross-Validation Results

| cv_rank | name | regressor | status | cv_folds | rmse_mean | rmse_std | mae_mean | mae_std | custom_score_mean | custom_score_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | elasticnet-a01-l03 | elasticnet | ok | 3 | 4.50964155801705 | 0.2121284677315245 | 3.432908924774507 | 0.1934982776214721 | 3.7559287147472697 | 0.19692455726443567 |
| 2 | elasticnet-default | elasticnet | ok | 3 | 4.539374053434883 | 0.15313044805714796 | 3.478908981392902 | 0.1541413785061562 | 3.7970485030054966 | 0.15179060393483737 |
| 3 | ridge-alpha-01 | ridge | ok | 3 | 5.131902115912163 | 0.37214511815816004 | 3.900689513485337 | 0.1366771927653801 | 4.270053294213385 | 0.19406659741955326 |
| 4 | ridge-alpha-1 | ridge | ok | 3 | 5.132086496222712 | 0.372123143418126 | 3.9008552701006374 | 0.13665824438082952 | 4.27022463793726 | 0.19404196911617008 |
| 5 | ridge-alpha-10 | ridge | ok | 3 | 5.1339303990904215 | 0.3719034006874418 | 3.9025109788279564 | 0.13646927606011686 | 4.271936804906695 | 0.19379598466441936 |

## Top Cross-Validation Predictions

- Test name: `split_0.1_round2`
- Folder: `Challenge1/output_data/test_top5/split_0.1_round2_cv_top5_predictions`
- Manifest: `Challenge1/output_data/test_top5/split_0.1_round2_cv_top5_predictions/manifest.csv`

| cv_rank | name | regressor | prediction_file |
| --- | --- | --- | --- |
| 1 | elasticnet-a01-l03 | elasticnet | cv_rank_01_elasticnet-a01-l03_y_pred.csv |
| 2 | elasticnet-default | elasticnet | cv_rank_02_elasticnet-default_y_pred.csv |
| 3 | ridge-alpha-01 | ridge | cv_rank_03_ridge-alpha-01_y_pred.csv |
| 4 | ridge-alpha-1 | ridge | cv_rank_04_ridge-alpha-1_y_pred.csv |
| 5 | ridge-alpha-10 | ridge | cv_rank_05_ridge-alpha-10_y_pred.csv |

