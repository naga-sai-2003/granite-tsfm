import warnings
import logging
import os
log_path = 'ttm.log'
if os.path.exists(log_path):
    os.remove(log_path)

logging.basicConfig(
    filename='ttm.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)

# redirect warnings to the logging system.
logging.captureWarnings(True)

# test warning and error messages.
warnings.warn('this is a warning message')
logging.error('this is an error message')

from eq_common_utils.utils.opensearch_helper import OpenSearch
from datetime import datetime, timedelta
import configparser
import os
import json
import copy
import pandas as pd
import traceback
from dotenv import load_dotenv

load_dotenv()

config=configparser.ConfigParser()
properties_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ttm.properties')
config.read(properties_path)

host = config.get(section='opensearch_prod', option='host')
port = config.get(section='opensearch_prod', option='port')
key_id = os.getenv('key_id')
secret  = os.getenv('secret')
region = config.get(section='opensearch_prod', option='region')
es = OpenSearch(host, port, key_id, secret, region)

def get_es_data(isin, dates, index_prefix, cols=None):
    try:
        data=[]
        for year in range(dates[0], dates[1]+1):
            q_total = '{"query":{"bool": {"must":[{"bool":{"should":[{"match":{"isin":"'+isin+'"}}]}}]}}}'
            if cols is not None:
                q_total = json.loads(q_total)
                q_total['_source'] = cols
                q_total = json.dumps(q_total)
            try:
                result=es.run_query(query=json.loads(q_total), index=f'{index_prefix}_{year}')
            except Exception as e:
                pass
            for rs in result['hits']['hits']:
                es_data=rs['_source']
                data.append(es_data)
        df=pd.DataFrame(data)
        df.sort_values(by='date', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if cols is not None:
            df = df[cols]
        return df
    except Exception as e:
        pass

def get_values_from_es(isin, date_needed, index, schedular, cols=None):
    year=pd.to_datetime(date_needed).date().year
    try:
        df = get_es_data(isin, [year,year], f'{index}', cols)
        df = df[df['schedular'] == f'{schedular}']
        df = df[pd.to_datetime(df['date']) == pd.to_datetime(date_needed)]
        return df
    except Exception as e:
        logging.error(f"data not available for isin: {isin} for year: {year}", e)