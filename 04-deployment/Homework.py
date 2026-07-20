import argparse
import os
import pickle

import pandas as pd

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def run(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output_{year:04d}-{month:02d}.parquet'

    df = read_data(input_file)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f'Q1 answer -- std dev of predicted duration: {y_pred.std():.2f}')
    print(f'Q5 answer -- mean predicted duration: {y_pred.mean():.2f}')

    # Q2: build ride_id and save results as parquet
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred,
    })

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    size_bytes = os.path.getsize(output_file)
    print(f'Q2 answer -- output file size: {size_bytes} bytes ({size_bytes / 1024 / 1024:.2f} MB)')

    print('Q3 answer -- command used to convert notebook to script: jupyter nbconvert --to script starter.ipynb')
    print('Q4 answer -- run `pipenv lock` then check the first hash under scikit-learn in Pipfile.lock')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    run(args.year, args.month)
