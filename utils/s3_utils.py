import boto3
from io import StringIO, BytesIO
import pandas as pd
import pickle
import gc
import os
import openpyxl
import configparser
import torch
from dotenv import load_dotenv

load_dotenv()

config=configparser.ConfigParser()
properties_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ttm.properties')
config.read(properties_path)

aws_access_key=os.getenv('aws_access_key_id')
aws_secret_key=os.getenv('aws_secret_key')
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

def s32df(bucket_name, file_name):
    s3_client = boto3.client('s3',aws_access_key_id=aws_access_key,
                             aws_secret_access_key=aws_secret_key)
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    csv_string = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))
    return df

def df2s3(df, bucket_name, file_name):
    session = boto3.Session(aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key)
    s3_res = session.resource('s3')                #resource is a high level feature in boto3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index = False)
    s3_res.Object(bucket_name, file_name).put(Body=csv_buffer.getvalue())
    return True

def get_excel_data_from_s3(bucket_name, object_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    body = obj['Body'].read()
    data = pd.read_excel(BytesIO(body), sheet_name='Sheet1')
    return data

def write_model_to_s3(model, bucket_name, file_name):
    s3_client=boto3.client('s3', aws_access_key_id = aws_access_key, aws_secret_access_key = aws_secret_key)

    with BytesIO() as model_buffer:
        torch.save(model.state_dict(), model_buffer)
        model_buffer.seek(0)
        s3_client.upload_fileobj(model_buffer, bucket_name, file_name)
    
    return True

def read_model_from_s3(model, bucket_name, file_name):
    s3_client=boto3.client('s3', aws_access_key_id = aws_access_key, aws_secret_access_key = aws_secret_key)

    with BytesIO() as model_buffer:
        s3_client.download_fileobj(bucket_name, file_name, model_buffer)
        model_buffer.seek(0)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        state_dict=torch.load(model_buffer, map_location=torch.device(device))
        model.load_state_dict(state_dict)
    
    return model

def upload_file_to_s3(local_path, bucket_name, s3_path):
    s3_client=boto3.client('s3', aws_access_key_id = aws_access_key, aws_secret_access_key = aws_secret_key)

    try:
        s3_client.upload_file(local_path, bucket_name, s3_path)
    except Exception as e:
        pass
    
