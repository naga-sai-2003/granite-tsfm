import requests
import pandas as pd
from utils import *

import math
import traceback
import warnings
warnings.filterwarnings("ignore")

# transformer-utilities.
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
import pandas as pd
from tqdm import tqdm

# tsfm-utilities.
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.callbacks import TrackingCallback
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# TODO: read the firms_url and tier1_url from the .properties.
firms_url = ' http://alb-master-1830456747.us-east-1.elb.amazonaws.com:8080/masteractivefirms/getmasterbytag?tag=aieq'
tier1_url = ' http://alb-master-1830456747.us-east-1.elb.amazonaws.com:8080/masteractivefirms/getmasterbytag?tag=tier1'
aieq_json = requests.get(f'{firms_url}').json()
aieq_companies = pd.DataFrame(aieq_json['data']['masteractivefirms_aieq'])
tier1_json = requests.get(f'{tier1_url}').json()
tier1_companies = pd.DataFrame(tier1_json['data']['masteractivefirms_tier1'])
aieq_isins = aieq_companies['isin'].tolist()

aieq_isins = aieq_companies['isin'].tolist()
experiments=pd.read_csv('experiments.csv')
exp=experiments.loc[1]


# TODO: read these parameters from the ttm.properties file.
# defining the ttm-hyper-parameters
output_directory='ttm_finetuned_models/'
ttm_model_revision='1024_96_v1'
context_length=int(exp['context_length'])
forecast_length=int(exp['forecast_length'])
fewshot_fraction=float(exp['fewshot_fraction'])
learning_rate = float(exp['learning_rate'])
num_epochs = 50
batch_size = int(exp['batch_size'])

# define the confidence score function.
def get_confidence_score(df, metrics_to_calculate=['confidence_score']):
    metrics_columns = metrics_to_calculate
    min_confidence = 0.01
    max_values = df['actual_monthly_return'].rolling(22 * 24, min_periods=22).max()
    min_values = df['actual_monthly_return'].rolling(22 * 24, min_periods=22).min()

    def calculate_confidence_score(row, i):
        if row['actual_monthly_return_predictions'] >= max_values.loc[i] or \
           row['actual_monthly_return_predictions'] <= min_values.loc[i] or \
           row['actual_monthly_return'] >= max_values.loc[i] or \
           row['actual_monthly_return'] <= min_values.loc[i]:
            return min_confidence
        else:
            if row['actual_monthly_return'] > row['actual_monthly_return_predictions']:
                return max(min_confidence, (max_values.loc[i] - row['actual_monthly_return']) / (max_values.loc[i] - row['actual_monthly_return_predictions']))
            else:
                return max(min_confidence, (row['actual_monthly_return'] - min_values.loc[i]) / (row['actual_monthly_return_predictions'] - min_values.loc[i]))

    if 'confidence_score' in metrics_columns or 'avg_confidence_score' in metrics_columns:
        return df.apply(calculate_confidence_score, axis=1, args=(range(len(df)),))


# input-data function.
def get_data(isin):
    # Get data from Elasticsearch
    data = get_es_data(isin, [2009, 2024], 'eq_cat_er_model')
    
    # filter data for monthly schedular.
    data = data[data['schedular'] == 'Monthly']
    
    # compute actual monthly return.
    data['actual_monthly_return'] = data['closeprice'].pct_change(periods=22)
    data['actual_monthly_return'] = data['actual_monthly_return'].shift(-22)
    

    data.rename(columns={'actual_monthly_return_predictions': 'er-predictions'}, inplace=True)
    # normalize the er-predicitions to avoid the over-flow in the ttm-model.
    data['er-predictions'] = data['er-predictions'] / 100

    cls = ['isin', 'date', 'actual_monthly_return', 'er-predictions']
    data = data[cls]
    
    # forward-fill for missing values.
    data = data.ffill()
    
    # save the data to s3.
    data_key = f'test/varaprasad-ttm-experiments/data/normalized-cat-er-forward-fill/{isin}.csv'
    df2s3(data, 'micro-ops-output', data_key)
    return data

def run(train_year, index):
    isins=aieq_isins[index : index + 100]
    for idx, isin in tqdm(enumerate(isins)):
        res_file=f'test/varaprasad-ttm-experiments/experiments/fewshot/aieq-cater-data-fill/historical/{train_year+1}/preds/univariate/{isin}.csv'
        try:
            s32df('micro-ops-output', res_file)
            continue
        except Exception as e:
            pass

        try:
            data=get_data(isin)
            data['date']=pd.to_datetime(data['date'], format='%Y-%m-%d')

            dataset_train=data[data['date'] <= f'{train_year}-06-30']
            dataset_valid=data[data['date'] > f'{train_year}-06-30']
            dataset_test=data[data['date'] > f'{train_year}-12-31']

            train_length = len(dataset_train)
            test_length = len(dataset_test)
            valid_length = len(dataset_valid)

            timestamp_column = 'date'
            id_columns = []
            target_columns = ['actual_monthly_return']

            split_configuration = {
                'train': [0, train_length],
                'valid': [train_length, len(data) - test_length],
                'test': [len(data) - test_length, len(data) - test_length]
            }

            column_specifiers = {
                'timestamp_column': timestamp_column,
                'id_columns': id_columns,
                'target_columns': target_columns,
                'control_columns': [],
            }
            
            tsp=TimeSeriesPreprocessor(
                **column_specifiers,
                context_length=context_length,
                prediction_length=forecast_length,
                scaling=False,
                encode_categorical=False,
                scaler_type='standard',
            )

            train_dataset, valid_dataset, test_dataset = tsp.get_datasets(
                data, split_configuration, fewshot_fraction=fewshot_fraction, fewshot_location='first'
            )

            forecast_model = TinyTimeMixerForPrediction.from_pretrained('ibm/TTM', revision=ttm_model_revision, head_dropout=float(exp['head_dropout_rate']))
            for param in forecast_model.backbone.parameters():
                param.requires_grad = not(bool(exp['backbone_freeze']))
            
            forecast_arguments = TrainingArguments(
                output_dir=os.path.join(output_directory, 'output'),
                overwrite_output_dir=True,
                learning_rate=learning_rate,
                num_train_epochs=num_epochs,
                do_eval=True,
                evaluation_strategy='epoch',
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                dataloader_num_workers=8,
                report_to=None,
                save_strategy='epoch',
                logging_strategy='epoch',
                save_total_limit=1,
                logging_dir=os.path.join(output_directory, 'logs'),
                load_best_model_at_end=True,
                metric_for_best_model='eval_loss',
                greater_is_better=False
            )

            early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.0,)
            tracking_callback = TrackingCallback()

            # optimizer and scheduler
            optimizer=AdamW(forecast_model.parameters(), lr=learning_rate)
            schedular=OneCycleLR(
                optimizer,
                learning_rate,
                epochs=num_epochs,
                steps_per_epoch=math.ceil(len(train_dataset)/(batch_size)),
            )

            forecast_trainer=Trainer(
                model=forecast_model,
                args=forecast_arguments,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[early_stopping_callback, tracking_callback],
                optimizers=(optimizer, schedular))

            # train the model.
            forecast_trainer.train()
            model_key=f'test/varaprasad-ttm-experiments/experiments/fewshot/aieq-cater-data-fill/historical/{train_year+1}/models/univariate/ttm_{isin}.bin'
            write_model_to_s3(forecast_trainer.model, 'micro-ops-output', model_key)

            res = forecast_trainer.evaluate(test_dataset)
            eval_loss = res['eval_loss']
            split_configuration = {
                'train': [0, train_length],
                'valid': [train_length, len(data) - test_length],
                'test': [len(data) - test_length, len(data)]
            }
            tsp_test = TimeSeriesPreprocessor(
                **column_specifiers,
                context_length=context_length,
                prediction_length=1,
                scaling=False,
                encode_categorical=False,
                scaler_type="standard",
            )
            train_dataset, valid_dataset, test_dataset = tsp_test.get_datasets(
                data, split_configuration, fewshot_fraction=fewshot_fraction, fewshot_location="first"
            )

            preds_fewshot = forecast_trainer.predict(test_dataset)
            predictions_fewshot = preds_fewshot[0][0]
            final_preds_fewshot = predictions_fewshot[:,0,:]
            final_preds_fewshot = final_preds_fewshot.flatten()
            dataset_test = data[-len(test_dataset):].reset_index().drop('index', axis=1)
            dataset_test['actual_monthly_return_predictions'] = final_preds_fewshot
            dataset_test['mse'] = (dataset_test['actual_monthly_return'] - dataset_test['actual_monthly_return_predictions']).rolling(22).apply(lambda x : ((x ** 2).sum()/22))
            dataset_test['rmse'] = (dataset_test['mse'] ** 0.5)
            dataset_test['mean_directionality'] = (dataset_test['actual_monthly_return'] * dataset_test['actual_monthly_return_predictions']).rolling(22).apply(lambda x: 100 * ((x) > 0).sum() / 22)
            confidence_scores = get_confidence_score(dataset_test.reset_index(drop = True))
            dataset_test['avg_confidence_score'] = pd.Series(confidence_scores).rolling(22 // 2).mean()
            df2s3(dataset_test, 'micro-ops-output', res_file) 

        except Exception as e:
            continue
    return True