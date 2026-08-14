#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  PYTHON=".venv/Scripts/python.exe"
else
  PYTHON="python"
fi

export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"
export AWS_DEFAULT_REGION="us-east-1"

docker compose up -d

echo "waiting for localstack to be healthy..."
until [ "$(docker inspect --format='{{.State.Health.Status}}' 06-best-practices-localstack-1 2>/dev/null)" = "healthy" ]; do
  sleep 1
done

"$PYTHON" -c "import awscli.clidriver, sys; sys.argv=['aws','--endpoint-url=http://localhost:4566','s3','mb','s3://nyc-duration']; sys.exit(awscli.clidriver.main())" || true

set +e
"$PYTHON" -m pytest -s tests/
pytest_result=$?

"$PYTHON" integration_test.py
integration_result=$?
set -e

docker compose down

echo ""
echo "Q1 answer: if statement looks like this: if __name__ == '__main__':"
echo "Q2 answer: The other file should be __init__.py"
echo "Q4 answer: --endpoint-url option is needed"

if [ ${pytest_result} -ne 0 ] || [ ${integration_result} -ne 0 ]; then
  exit 1
fi
