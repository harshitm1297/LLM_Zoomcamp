import os

import logfire.db_api
from dotenv import load_dotenv


load_dotenv('../.env')

with logfire.db_api.connect(
    read_token=os.environ['LOGFIRE_READ_TOKEN']
) as connection:
    cursor = connection.cursor()
    cursor.execute(
        '''
        WITH latest_agent_trace AS (
            SELECT trace_id
            FROM records
            WHERE span_name = 'agent run'
               OR message = 'faq_agent run'
            ORDER BY start_timestamp DESC
            LIMIT 1
        )
        SELECT span_name, message, kind
        FROM records
        WHERE trace_id = (SELECT trace_id FROM latest_agent_trace)
        ORDER BY start_timestamp
        '''
    )
    rows = cursor.fetchall()

print(f'span_count={len(rows)}')
for row in rows:
    print(row)
