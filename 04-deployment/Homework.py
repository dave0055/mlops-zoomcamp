import argparse
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
    print('Q4 -- sha256:057b991ac64b3e75c9c04b5f9395eaf19a6179244c089afdebaad98264bff37c')
    print(f'Q5 -- mean predicted duration: {y_pred.mean():.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    run(args.year, args.month)
