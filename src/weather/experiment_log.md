  Models (5-fold CV, 80 features):
  Model                    Accuracy
  ---------------------- ----------
  LogisticRegression         67.2%filterair fi
  RandomForest               73.4%
  HistGBM                    72.1%

WITH ALL SITES

    Models (5-fold CV, 80 features):
  Model                    Accuracy
  ---------------------- ----------
  LogisticRegression         68.2%
  RandomForest               76.1%
  HistGBM                    78.1%


sin/cos of day are being flagged as important features. No idea why, this should not be the case since they shouldnt help with bracket information



==============================================================================
  HISTGBM HYPERPARAMETER TUNING (19941 rows, 78 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.1, 'max_depth': 7, 'max_iter': 100, 'min_samples_leaf': 10}
  Best CV accuracy: 74.7%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     74.7%       200     7   0.10       10
     1     74.7%       100     7   0.10       10
     1     74.7%       400     7   0.10       10
     4     74.7%       100     7   0.05        5
     5     74.7%       100     7   0.05       20
     6     74.7%       100     7   0.05       10
     7     74.7%       400     7   0.05        5
     7     74.7%       200     7   0.05        5
     9     74.7%       400     7   0.05       10
     9     74.7%       200     7   0.05       10

       Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     74.8%       400    12   0.05       20
     1     74.8%       200    12   0.05       20
     3     74.8%       100    12   0.05       20
     4     74.7%       200     7   0.10       10
     4     74.7%       100     7   0.10       10
     4     74.7%       400     7   0.10       10
     7     74.7%       100     7   0.05       20
     8     74.7%       100     7   0.05       10
     9     74.7%       400     9   0.05       40
     9     74.7%       200     9   0.05       40


    

  Model                     Exact    Upper   Middle
  ---------------------- -------- -------- --------
  S1 (auto-only)           71.3%   82.2%   80.6%
  S2 (S1+METAR)            72.8%   82.6%   81.5%



  FITTING FOR AUTO FEATURES ONLY

  ==============================================================================
  HISTGBM HYPERPARAMETER TUNING (19941 rows, 69 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.05, 'max_depth': 9, 'max_iter': 100, 'min_samples_leaf': 40}
  Best CV accuracy: 71.6%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     71.6%       100     9   0.05       40
     2     71.6%       100     5   0.10       10
     2     71.6%       200     5   0.10       10
     2     71.6%       400     5   0.10       10
     5     71.6%       100     9   0.10       20
     5     71.6%       400     9   0.10       20
     5     71.6%       200     9   0.10       20
     8     71.6%       400     7   0.05       40
     8     71.6%       200     7   0.05       40
    10     71.6%       400     9   0.10       40




    EXP LOG 3/20 -> VALDIATE ON 1/1/2026+ AND ADD MAX_C FEATURE

    (base) michael@DESKTOP-NBFMBQF:~/workspace/polymarket-trader$ python src/weather/backtest_rounding.py --regression  --stage1-only --tune --use-all-sites
==============================================================================
  HISTGBM HYPERPARAMETER TUNING (19765 rows, 72 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.05, 'max_depth': 7, 'max_iter': 200, 'min_samples_leaf': 40}
  Best CV accuracy: 72.7%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     72.7%       400     7   0.05       40
     1     72.7%       400    12   0.05       40
     1     72.7%       200    12   0.05       40
     1     72.7%       100    12   0.05       40
     1     72.7%       200     7   0.05       40
     6     72.7%       100     7   0.05       40
     7     72.7%       400     7   0.05       10
     7     72.7%       200     7   0.05       10
     9     72.7%       100     7   0.05       10
    10     72.7%       200    12   0.05       10
(base) michael@DESKTOP-NBFMBQF:~/workspace/polymarket-trader$ python src/weather/backtest_rounding.py --regression  --stage1-only --tune
==============================================================================
  HISTGBM HYPERPARAMETER TUNING (3038 rows, 72 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.05, 'max_depth': 7, 'max_iter': 100, 'min_samples_leaf': 40}
  Best CV accuracy: 70.5%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     70.5%       400     7   0.05       40
     1     70.5%       200     7   0.05       40
     1     70.5%       100     7   0.05       40
     4     70.4%       400     7   0.10       40
     4     70.4%       100     7   0.10       40
     4     70.4%       200     7   0.10       40
     7     70.3%       200     9   0.05       40
     7     70.3%       100     9   0.05       40
     7     70.3%       400     9   0.05       40
    10     70.3%       100     5   0.20       40



what the fuk is this??? weather data is corrupted before 11 or something

(base) michael@DESKTOP-NBFMBQF:~/workspace/polymarket-trader$ python src/weather/backtest_rounding.py --regression  --stage1-only --tune
==============================================================================
  HISTGBM HYPERPARAMETER TUNING (3038 rows, 72 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.05, 'max_depth': 7, 'max_iter': 100, 'min_samples_leaf': 40}
  Best CV accuracy: 70.5%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     70.5%       400     7   0.05       40
     1     70.5%       200     7   0.05       40
     1     70.5%       100     7   0.05       40
     4     70.4%       400     7   0.10       40
     4     70.4%       100     7   0.10       40
     4     70.4%       200     7   0.10       40
     7     70.3%       200     9   0.05       40
     7     70.3%       100     9   0.05       40
     7     70.3%       400     9   0.05       40
    10     70.3%       100     5   0.20       40

(base) michael@DESKTOP-NBFMBQF:~/workspace/polymarket-trader$ python src/weather/backtest_rounding.py --stage1-only --tune --since 2026-01-01
  Filtered to dates >= 2026-01-01: 1282 rows
==============================================================================
  HISTGBM HYPERPARAMETER TUNING (1278 rows, 72 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.05, 'max_depth': 5, 'max_iter': 100, 'min_samples_leaf': 20}
  Best CV accuracy: 75.8%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     75.8%       100     5   0.05       20
     1     75.8%       200     5   0.05       20
     1     75.8%       400     5   0.05       20
     4     75.4%       100     5   0.05       10
     4     75.4%       200     5   0.05       10
     4     75.4%       400     5   0.05       10
     7     75.4%       400    12   0.10       10
     7     75.4%       200    12   0.10       10
     7     75.4%       100    12   0.10       10
    10     75.4%       100     5   0.10       20
(base) michael@DESKTOP-NBFMBQF:~/workspace/polymarket-trader$ python src/weather/backtest_rounding.py --stage1-only --tune --since 2025-11-01
  Filtered to dates >= 2025-11-01: 2296 rows
==============================================================================
  HISTGBM HYPERPARAMETER TUNING (2292 rows, 72 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.05, 'max_depth': 5, 'max_iter': 100, 'min_samples_leaf': 40}
  Best CV accuracy: 76.0%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     76.0%       100     5   0.05       40
     2     76.0%       100     7   0.05       20
     2     76.0%       200     7   0.05       20
     2     76.0%       400     7   0.05       20
     5     75.9%       400     7   0.10       20
     5     75.9%       200     7   0.10       20
     5     75.9%       100     7   0.10       20
     8     75.9%       200    12   0.05       20
     8     75.9%       100    12   0.05       20
     8     75.9%       400    12   0.05       20


     ALL SITES

     (base) michael@DESKTOP-NBFMBQF:~/workspace/polymarket-trader$ python src/weather/backtest_rounding.py --use-all-sites --stage1-only --tune --since 2025-11-01
  Filtered to dates >= 2025-11-01: 14997 rows
==============================================================================
  HISTGBM HYPERPARAMETER TUNING (14968 rows, 72 features)
==============================================================================
Fitting 5 folds for each of 108 candidates, totalling 540 fits

  Best params: {'learning_rate': 0.05, 'max_depth': 7, 'max_iter': 200, 'min_samples_leaf': 40}
  Best CV accuracy: 79.5%

  Top 10 configurations:
  Rank  Accuracy  max_iter depth     lr min_leaf
  ---- --------- --------- ----- ------ --------
     1     79.5%       400     7   0.05       40
     1     79.5%       200     7   0.05       40
     3     79.5%       100     7   0.05       40
     4     79.4%       100     5   0.05       20
     5     79.4%       400     7   0.05       10
     5     79.4%       200     7   0.05       10
     5     79.4%       100     7   0.05       10
     8     79.4%       100     7   0.05       20
     9     79.4%       200    12   0.05       10
     9     79.4%       100    12   0.05       10


fixed validation

(base) michael@DESKTOP-NBFMBQF:~/workspace/polymarket-trader$ python src/weather/backtest_rounding.py --verify-model 
  Loaded model: 2026-03-20T19:31:53.574477 (14968 rows, 113 sites)
  Pipeline: full 2-stage
  Since: 2026-01-01

==============================================================================
  BRACKET MODEL VERIFICATION
==============================================================================
  Total days: 1282  |  Sites: 17
    offset=-1:   318 ( 24.8%)
    offset=+0:   586 ( 45.7%)
    offset=+1:   378 ( 29.5%)

  Metric         Accuracy   Baseline     Lift
  ------------ ---------- ---------- --------
  Exact             86.9%      45.7%   +41.2%
  Upper             93.9%      75.2%   +18.7%
  Middle            91.6%      70.5%   +21.1%

  Per-offset accuracy:
    Offset      N  Correct     Acc
  -------- ------ -------- -------
        -1    318      277   87.1%
        +0    586      520   88.7%
        +1    378      317   83.9%

  Confusion matrix (rows=actual, cols=predicted):
            pred-1  pred 0  pred+1
  actual-1     277      30      11
  actual+0      32     520      34
  actual+1       7      54     317

  Per-site accuracy:
  Site         N   Exact   Upper  Middle    A0%
  -------- ----- ------- ------- ------- ------
  KATL        76   88.2%   94.7%   93.4%  44.7%
  KAUS        76   86.8%   94.7%   90.8%  44.7%
  KBOS        76   88.2%   92.1%   96.1%  48.7%
  KDCA        76   86.8%   93.4%   93.4%  50.0%
  KDEN        76   89.5%   97.4%   92.1%  39.5%
  KDFW        76   82.9%   93.4%   88.2%  42.1%
  KHOU        74   85.1%   93.2%   93.2%  51.4%
  KLAX        76   82.9%   93.4%   88.2%  39.5%
  KMDW        76   89.5%   92.1%   96.1%  51.3%
  KMIA        76   92.1%   96.1%   93.4%  48.7%
  KMSP        76   82.9%   90.8%   92.1%  44.7%
  KOKC        76   88.2%   94.7%   89.5%  48.7%
  KPHL        68   91.2%   95.6%   91.2%  38.2%
  KPHX        76   86.8%   96.1%   90.8%  51.3%
  KSAT        76   93.4%   98.7%   93.4%  40.8%
  KSEA        76   84.2%   90.8%   89.5%  38.2%
  KSFO        76   78.9%   89.5%   85.5%  53.9%

  Avg predicted prob for correct offset: 0.809



3/23/2026

adding new temp featurs


==============================================================================
  LEAVE-ONE-SITE-OUT (stage1-only)
  Training sites (96) always in training set
  Test set: 8412 days from 2026-01-01 onward
==============================================================================
  Site         N    A0%   Exact   Upper  Middle
  -------- ----- ------ ------- ------- -------
  KATL        76 44.7%  76.3%  88.2%  86.8%
  KAUS        76 44.7%  73.7%  86.8%  84.2%
  KBOS        76 48.7%  80.3%  88.2%  92.1%
  KDCA        76 50.0%  73.7%  86.8%  82.9%
  KDEN        76 39.5%  75.0%  88.2%  82.9%
  KDFW        76 42.1%  78.9%  88.2%  86.8%
  KHOU        74 51.4%  75.7%  83.8%  89.2%
  KLAX        76 39.5%  75.0%  89.5%  81.6%
  KMDW        76 51.3%  80.3%  86.8%  89.5%
  KMIA        76 48.7%  85.5%  90.8%  92.1%
  KMSP        76 44.7%  75.0%  86.8%  85.5%
  KOKC        76 48.7%  82.9%  92.1%  88.2%
  KPHL        68 38.2%  80.9%  88.2%  86.8%
  KPHX        76 51.3%  73.7%  86.8%  85.5%
  KSAT        76 40.8%  86.8%  96.1%  90.8%
  KSEA        76 38.2%  80.3%  86.8%  89.5%
  KSFO        76 53.9%  72.4%  82.9%  81.6%