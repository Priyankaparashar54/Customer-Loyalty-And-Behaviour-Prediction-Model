import datetime
import time
import warnings
import gc

from contextlib import contextmanager

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# Calculate RMSE
def rmse(y_true, y_predicted):
    return np.sqrt(mean_squared_error(y_true, y_predicted))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Display/plot feature importance
def display_importances(df_feat_import_, out_viz_dir):
    cols = df_feat_import_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance",
        ascending=False)[:40].index
    best_features = df_feat_import_.loc[df_feat_import_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(f'{out_viz_dir}/lgbm_importances.png')
    return


# Helper function to reduce memory of DataFrame
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    if verbose:
        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# date interval range function
def date_interval_range(x):
    if x < -7:
        x = -7
    elif x > 91:
        x = 91
    return x


# Feature engineering for train & test
def fe_train_test(fp_train, fp_test, num_rows=None):
    # load csv
    df_train = pd.read_csv(fp_train, index_col=['card_id'], nrows=num_rows)
    df_test = pd.read_csv(fp_test, index_col=['card_id'], nrows=num_rows)

    print("Train samples: {}, test samples: {}".format(len(df_train), len(df_test)))

    # outlier
    df_train['outliers'] = 0
    df_train.loc[df_train['target'] < -30, 'outliers'] = 1

    # set target as nan
    df_test['target'] = np.nan

    # merge
    df_train_test = df_train.append(df_test)

    del df_train, df_test
    gc.collect()

    # to datetime
    df_train_test['first_active_month'] = pd.to_datetime(df_train_test['first_active_month'])

    # datetime features
    df_train_test['quarter'] = df_train_test['first_active_month'].dt.quarter
    df_train_test['elapsed_time'] = (datetime.datetime.today() - df_train_test['first_active_month']).dt.days

    df_train_test['days_feat1'] = df_train_test['elapsed_time'] * df_train_test['feature_1']
    df_train_test['days_feat2'] = df_train_test['elapsed_time'] * df_train_test['feature_2']
    df_train_test['days_feat3'] = df_train_test['elapsed_time'] * df_train_test['feature_3']

    df_train_test['days_feat1_ratio'] = df_train_test['feature_1'] / df_train_test['elapsed_time']
    df_train_test['days_feat2_ratio'] = df_train_test['feature_2'] / df_train_test['elapsed_time']
    df_train_test['days_feat3_ratio'] = df_train_test['feature_3'] / df_train_test['elapsed_time']

    # one hot encoding
    df_train_test, cols = one_hot_encoder(df_train_test, nan_as_category=False)

    for f in ['feature_1', 'feature_2', 'feature_3']:
        order_label = df_train_test.groupby([f])['outliers'].mean()
        df_train_test[f] = df_train_test[f].map(order_label)

    df_train_test['feature_sum'] = df_train_test['feature_1'] + df_train_test['feature_2'] + df_train_test['feature_3']
    df_train_test['feature_mean'] = df_train_test['feature_sum'] / 3
    df_train_test['feature_max'] = df_train_test[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df_train_test['feature_min'] = df_train_test[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df_train_test['feature_var'] = df_train_test[['feature_1', 'feature_2', 'feature_3']].std(axis=1)

    return df_train_test


# Feature engineering for historical transactions
def fe_historic(fp_historic, num_rows=None):
    # load csv
    df_historic = pd.read_csv(fp_historic, nrows=num_rows)

    # reduce memory usage
    df_historic = reduce_mem_usage(df_historic)

    # fillna
    df_historic['category_2'].fillna(1.0, inplace=True)
    df_historic['category_3'].fillna('A', inplace=True)
    df_historic['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    df_historic['installments'].replace(-1, np.nan, inplace=True)
    df_historic['installments'].replace(999, np.nan, inplace=True)

    # trim
    df_historic['purchase_amount'] = df_historic['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    df_historic['authorized_flag'] = df_historic['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    df_historic['category_1'] = df_historic['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    df_historic['category_3'] = df_historic['category_3'].map({'A': 0, 'B': 1, 'C': 2})

    # datetime features
    df_historic['purchase_date'] = pd.to_datetime(df_historic['purchase_date'])
    df_historic['month'] = df_historic['purchase_date'].dt.month
    df_historic['day'] = df_historic['purchase_date'].dt.day
    df_historic['hour'] = df_historic['purchase_date'].dt.hour
    df_historic['weekofyear'] = df_historic['purchase_date'].dt.weekofyear
    df_historic['weekday'] = df_historic['purchase_date'].dt.weekday
    df_historic['weekend'] = (df_historic['purchase_date'].dt.weekday >= 5).astype(int)

    # additional features
    df_historic['price'] = df_historic['purchase_amount'] / df_historic['installments']

    # Christmas : December 25 2017
    df_historic['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - df_historic['purchase_date']).dt.days.apply(
        date_interval_range)
    # Mothers Day: May 14 2017
    df_historic['Mothers_Day_2017'] = (pd.to_datetime('2017-06-04') - df_historic['purchase_date']).dt.days.apply(
        date_interval_range)
    # fathers day: August 13 2017
    df_historic['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - df_historic['purchase_date']).dt.days.apply(
        date_interval_range)
    # Childrens day: October 12 2017
    df_historic['Children_day_2017'] = (pd.to_datetime('2017-10-12') - df_historic['purchase_date']).dt.days.apply(
        date_interval_range)
    # Valentine's Day : 12th June, 2017
    df_historic['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - df_historic['purchase_date']).dt.days.apply(
        date_interval_range)
    # Black Friday : 24th November 2017
    df_historic['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - df_historic['purchase_date']).dt.days.apply(
        date_interval_range)
    # Mothers Day: May 13 2018
    df_historic['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - df_historic['purchase_date']).dt.days.apply(
        date_interval_range)

    df_historic['month_diff'] = (datetime.datetime.today() - df_historic['purchase_date']).dt.days // 30
    df_historic['month_diff'] += df_historic['month_lag']

    # additional features
    df_historic['duration'] = df_historic['purchase_amount'] * df_historic['month_diff']
    df_historic['amount_month_ratio'] = df_historic['purchase_amount'] / df_historic['month_diff']

    # reduce memory usage
    df_historic = reduce_mem_usage(df_historic)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['authorized_flag'] = ['mean']
    aggs['weekend'] = ['mean']  # overwrite
    aggs['weekday'] = ['mean']  # overwrite
    aggs['day'] = ['nunique', 'mean', 'min']  # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['sum', 'mean', 'max', 'min', 'var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration'] = ['mean', 'min', 'max', 'var', 'skew']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

    for col in ['category_2', 'category_3']:
        df_historic[col + '_mean'] = df_historic.groupby([col])['purchase_amount'].transform('mean')
        df_historic[col + '_min'] = df_historic.groupby([col])['purchase_amount'].transform('min')
        df_historic[col + '_max'] = df_historic.groupby([col])['purchase_amount'].transform('max')
        df_historic[col + '_sum'] = df_historic.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    df_historic = df_historic.reset_index().groupby('card_id').agg(aggs)

    # change column name
    df_historic.columns = pd.Index([e[0] + "_" + e[1] for e in df_historic.columns.tolist()])
    df_historic.columns = ['hist_' + c for c in df_historic.columns]

    df_historic['hist_purchase_date_diff'] = (
            df_historic['hist_purchase_date_max'] - df_historic['hist_purchase_date_min']).dt.days
    df_historic['hist_purchase_date_average'] = (
            df_historic['hist_purchase_date_diff'] / df_historic['hist_card_id_size'])
    df_historic['hist_purchase_date_uptonow'] = (
            datetime.datetime.today() - df_historic['hist_purchase_date_max']).dt.days
    df_historic['hist_purchase_date_uptomin'] = (
            datetime.datetime.today() - df_historic['hist_purchase_date_min']).dt.days

    # reduce memory usage
    df_historic = reduce_mem_usage(df_historic)

    return df_historic


# Feature engineering for new_merchant_transactions
def fe_new(fp_new, num_rows=None):
    # load csv
    df_new = pd.read_csv(fp_new, nrows=num_rows)

    # fillna
    df_new['category_2'].fillna(1.0, inplace=True)
    df_new['category_3'].fillna('A', inplace=True)
    df_new['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
    df_new['installments'].replace(-1, np.nan, inplace=True)
    df_new['installments'].replace(999, np.nan, inplace=True)

    # trim
    df_new['purchase_amount'] = df_new['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    df_new['authorized_flag'] = df_new['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    df_new['category_1'] = df_new['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    df_new['category_3'] = df_new['category_3'].map({'A': 0, 'B': 1, 'C': 2}).astype(int)

    # datetime features
    df_new['purchase_date'] = pd.to_datetime(df_new['purchase_date'])
    df_new['month'] = df_new['purchase_date'].dt.month
    df_new['day'] = df_new['purchase_date'].dt.day
    df_new['hour'] = df_new['purchase_date'].dt.hour
    df_new['weekofyear'] = df_new['purchase_date'].dt.weekofyear
    df_new['weekday'] = df_new['purchase_date'].dt.weekday
    df_new['weekend'] = (df_new['purchase_date'].dt.weekday >= 5).astype(int)

    # additional features
    df_new['price'] = df_new['purchase_amount'] / df_new['installments']

    # Christmas : December 25 2017
    df_new['Christmas_Day_2017'] = (
                pd.to_datetime('2017-12-25') - df_new['purchase_date']).dt.days.apply(date_interval_range)
    # Childrens day: October 12 2017
    df_new['Children_day_2017'] = (
                pd.to_datetime('2017-10-12') - df_new['purchase_date']).dt.days.apply(date_interval_range)
    # Black Friday : 24th November 2017
    df_new['Black_Friday_2017'] = (
                pd.to_datetime('2017-11-24') - df_new['purchase_date']).dt.days.apply(date_interval_range)
    # Mothers Day: May 13 2018
    df_new['Mothers_Day_2018'] = (
                pd.to_datetime('2018-05-13') - df_new['purchase_date']).dt.days.apply(date_interval_range)

    df_new['month_diff'] = (datetime.datetime.today() - df_new['purchase_date']).dt.days // 30
    df_new['month_diff'] += df_new['month_lag']

    # additional features
    df_new['duration'] = df_new['purchase_amount'] * df_new['month_diff']
    df_new['amount_month_ratio'] = df_new['purchase_amount'] / df_new['month_diff']

    # reduce memory usage
    df_new = reduce_mem_usage(df_new)

    col_unique = ['subsector_id', 'merchant_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum', 'max', 'min', 'mean', 'var', 'skew']
    aggs['installments'] = ['sum', 'max', 'mean', 'var', 'skew']
    aggs['purchase_date'] = ['max', 'min']
    aggs['month_lag'] = ['max', 'min', 'mean', 'var', 'skew']
    aggs['month_diff'] = ['mean', 'var', 'skew']
    aggs['weekend'] = ['mean']
    aggs['month'] = ['mean', 'min', 'max']
    aggs['weekday'] = ['mean', 'min', 'max']
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['size', 'count']
    aggs['price'] = ['mean', 'max', 'min', 'var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration'] = ['mean', 'min', 'max', 'var', 'skew']
    aggs['amount_month_ratio'] = ['mean', 'min', 'max', 'var', 'skew']

    for col in ['category_2', 'category_3']:
        df_new[col + '_mean'] = df_new.groupby([col])['purchase_amount'].transform('mean')
        df_new[col + '_min'] = df_new.groupby([col])['purchase_amount'].transform('min')
        df_new[col + '_max'] = df_new.groupby([col])['purchase_amount'].transform('max')
        df_new[col + '_sum'] = df_new.groupby([col])['purchase_amount'].transform('sum')
        aggs[col + '_mean'] = ['mean']

    df_new = df_new.reset_index().groupby('card_id').agg(aggs)

    # change column name
    df_new.columns = pd.Index([e[0] + "_" + e[1] for e in df_new.columns.tolist()])
    df_new.columns = ['new_' + c for c in df_new.columns]

    df_new['new_purchase_date_diff'] = (
            df_new['new_purchase_date_max'] - df_new['new_purchase_date_min']).dt.days
    df_new['new_purchase_date_average'] = (
            df_new['new_purchase_date_diff'] / df_new['new_card_id_size'])
    df_new['new_purchase_date_uptonow'] = (
            datetime.datetime.today() - df_new['new_purchase_date_max']).dt.days
    df_new['new_purchase_date_uptomin'] = (
            datetime.datetime.today() - df_new['new_purchase_date_min']).dt.days

    # reduce memory usage
    df_new = reduce_mem_usage(df_new)

    return df_new


# Feature engineering for additional features
def fe_add(df_add):
    df_add['hist_first_buy'] = (df_add['hist_purchase_date_min'] - df_add['first_active_month']).dt.days
    df_add['hist_last_buy'] = (df_add['hist_purchase_date_max'] - df_add['first_active_month']).dt.days
    df_add['new_first_buy'] = (df_add['new_purchase_date_min'] - df_add['first_active_month']).dt.days
    df_add['new_last_buy'] = (df_add['new_purchase_date_max'] - df_add['first_active_month']).dt.days

    date_features = ['hist_purchase_date_max', 'hist_purchase_date_min',
                     'new_purchase_date_max', 'new_purchase_date_min']

    for f in date_features:
        df_add[f] = df_add[f].astype(np.int64) * 1e-9

    df_add['card_id_total'] = df_add['new_card_id_size'] + df_add['hist_card_id_size']
    df_add['card_id_cnt_total'] = df_add['new_card_id_count'] + df_add['hist_card_id_count']
    df_add['card_id_cnt_ratio'] = df_add['new_card_id_count'] / df_add['hist_card_id_count']
    df_add['purchase_amount_total'] = df_add['new_purchase_amount_sum'] + df_add['hist_purchase_amount_sum']
    df_add['purchase_amount_mean'] = df_add['new_purchase_amount_mean'] + df_add['hist_purchase_amount_mean']
    df_add['purchase_amount_max'] = df_add['new_purchase_amount_max'] + df_add['hist_purchase_amount_max']
    df_add['purchase_amount_min'] = df_add['new_purchase_amount_min'] + df_add['hist_purchase_amount_min']
    df_add['purchase_amount_ratio'] = df_add['new_purchase_amount_sum'] / df_add['hist_purchase_amount_sum']
    df_add['month_diff_mean'] = df_add['new_month_diff_mean'] + df_add['hist_month_diff_mean']
    df_add['month_diff_ratio'] = df_add['new_month_diff_mean'] / df_add['hist_month_diff_mean']
    df_add['month_lag_mean'] = df_add['new_month_lag_mean'] + df_add['hist_month_lag_mean']
    df_add['month_lag_max'] = df_add['new_month_lag_max'] + df_add['hist_month_lag_max']
    df_add['month_lag_min'] = df_add['new_month_lag_min'] + df_add['hist_month_lag_min']
    df_add['category_1_mean'] = df_add['new_category_1_mean'] + df_add['hist_category_1_mean']
    df_add['installments_total'] = df_add['new_installments_sum'] + df_add['hist_installments_sum']
    df_add['installments_mean'] = df_add['new_installments_mean'] + df_add['hist_installments_mean']
    df_add['installments_max'] = df_add['new_installments_max'] + df_add['hist_installments_max']
    df_add['installments_ratio'] = df_add['new_installments_sum'] / df_add['hist_installments_sum']
    df_add['price_total'] = df_add['purchase_amount_total'] / df_add['installments_total']
    df_add['price_mean'] = df_add['purchase_amount_mean'] / df_add['installments_mean']
    df_add['price_max'] = df_add['purchase_amount_max'] / df_add['installments_max']
    df_add['duration_mean'] = df_add['new_duration_mean'] + df_add['hist_duration_mean']
    df_add['duration_min'] = df_add['new_duration_min'] + df_add['hist_duration_min']
    df_add['duration_max'] = df_add['new_duration_max'] + df_add['hist_duration_max']
    df_add['amount_month_ratio_mean'] = df_add['new_amount_month_ratio_mean'] + df_add['hist_amount_month_ratio_mean']
    df_add['amount_month_ratio_min'] = df_add['new_amount_month_ratio_min'] + df_add['hist_amount_month_ratio_min']
    df_add['amount_month_ratio_max'] = df_add['new_amount_month_ratio_max'] + df_add['hist_amount_month_ratio_max']
    df_add['new_CLV'] = df_add['new_card_id_count'] * df_add['new_purchase_amount_sum'] / df_add['new_month_diff_mean']
    df_add['hist_CLV'] = \
        df_add['hist_card_id_count'] * df_add['hist_purchase_amount_sum'] / df_add['hist_month_diff_mean']
    df_add['CLV_ratio'] = df_add['new_CLV'] / df_add['hist_CLV']

    return df_add


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df_train, df_test, num_folds=5, drop_feats=None, stratified=False, out_dir_data=None,
                   out_dir_viz=None):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(df_train.shape, df_test.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    df_feat_import = pd.DataFrame()
    drop_feats = [] if drop_feats is None else drop_feats
    feats = [f for f in df_train.columns if f not in drop_feats]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['outliers'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['target'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params optimized by optunagh
        params = {
            'task': 'train',
            'boosting': 'goss',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'top_rate': 0.9064148448434349,
            'num_leaves': 63,
            'min_child_weight': 41.9612869171337,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.677537745007898,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 9.820197773625843,
            'reg_lambda': 8.2532317400459,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed': int(2 ** n_fold),
            'bagging_seed': int(2 ** n_fold),
            'drop_seed': int(2 ** n_fold)
        }

        reg = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=['train', 'test'],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=100
        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(df_test[feats], num_iteration=reg.best_iteration) / folds.n_splits

        df_fold_import = pd.DataFrame()
        df_fold_import["feature"] = feats
        df_fold_import["importance"] = np.log1p(
            reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        df_fold_import["fold"] = n_fold + 1
        df_feat_import = pd.concat([df_feat_import, df_fold_import], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # display importances
    if out_dir_viz:
        display_importances(df_feat_import, out_dir_viz)

    if out_dir_data:
        submission_fn = '{}/submission.csv'.format(out_dir_data)
        df_test.loc[:, 'target'] = sub_preds
        df_test = df_test.reset_index()
        df_test[['card_id', 'target']].to_csv(submission_fn, index=False)
        print('Test predictions saved as {}'.format(submission_fn))

    return
