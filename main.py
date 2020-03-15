import gc

from pandas import merge
from data import timer, fe_train_test, fe_historic, fe_new, fe_add, kfold_lightgbm


def main(debug=False):
    num_rows = 10000 if debug else None
    dir_data = 'data'
    drop_feats = [
        'first_active_month', 'target', 'card_id', 'outliers', 'hist_purchase_date_max', 'hist_purchase_date_min',
        'hist_card_id_size', 'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size', 'OOF_PRED', 'month_0'
    ]
    with timer("train & test"):
        df = fe_train_test(f'{dir_data}/train.csv', f'{dir_data}/test.csv', num_rows)
    with timer("historical transactions"):
        df = merge(df, fe_historic(f'{dir_data}/historical_transactions.csv', num_rows), on='card_id', how='outer')
    with timer("new merchants"):
        df = merge(df, fe_new(f'{dir_data}/new_merchant_transactions.csv', num_rows), on='card_id', how='outer')
    with timer("additional features"):
        df = fe_add(df)
    with timer("split train & test"):
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        del df
        gc.collect()
    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(
            df_train=train_df,
            df_test=test_df,
            num_folds=11,
            drop_feats=drop_feats,
            stratified=False,
            out_dir_data='data',
            out_dir_viz='vizualizations'
        )


if __name__ == "__main__":
    with timer("Full model run"):
        main(debug=False)
