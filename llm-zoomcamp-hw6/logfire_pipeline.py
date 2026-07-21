import os
from datetime import datetime, timezone

import dlt
from dotenv import load_dotenv
from logfire.experimental.query_client import LogfireQueryClient


load_dotenv('../.env')

@dlt.resource(name='records', write_disposition='replace')
def logfire_records():
    read_token = os.environ['LOGFIRE_READ_TOKEN']
    with LogfireQueryClient(read_token=read_token) as client:
        result = client.query_json_rows(
            sql='''
                SELECT *
                FROM records
                WHERE trace_id IN (
                    SELECT trace_id
                    FROM records
                    WHERE message = 'faq_agent run'
                      AND NOT is_exception
                )
                ORDER BY start_timestamp
            ''',
            min_timestamp=datetime(1970, 1, 1, tzinfo=timezone.utc),
            limit=10_000,
        )
    yield from result['rows']


def main() -> None:
    pipeline = dlt.pipeline(
        pipeline_name='agent_traces_pipeline',
        destination='duckdb',
        dataset_name='agent_traces',
    )
    load_info = pipeline.run(logfire_records())
    print(load_info)
    print(f'DuckDB path: {pipeline.pipeline_name}.duckdb')


if __name__ == '__main__':
    main()
