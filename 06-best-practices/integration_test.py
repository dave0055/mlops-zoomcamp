import io
import os
import sys
from datetime import datetime

import pandas as pd

os.environ.setdefault('AWS_ACCESS_KEY_ID', 'test')
os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'test')
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
os.environ['S3_ENDPOINT_URL'] = 'http://localhost:4566'
os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
os.environ['OUTPUT_FILE_PATTERN'] = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'

import batch


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def main():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    year = 2023
    month = 1

    input_file = batch.get_input_path(year, month)
    output_file = batch.get_output_path(year, month)

    options = batch.get_storage_options()
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options,
    )
    print(f'saved input to {input_file}')

    buffer = io.BytesIO()
    df_input.to_parquet(buffer, engine='pyarrow', compression=None, index=False)
    print(f'Q5 answer: The size of the file is {buffer.tell()}')

    exit_code = os.system(f'"{sys.executable}" batch.py {year} {month}')
    assert exit_code == 0

    df_result = pd.read_parquet(output_file, storage_options=options)
    print(f'read output from {output_file}')

    total_predicted_duration = df_result['predicted_duration'].sum()
    print('sum of predicted durations:', total_predicted_duration)
    print(f'Q6 answer: The sum of predicted duration is {total_predicted_duration:.2f}')


if __name__ == '__main__':
    main()
