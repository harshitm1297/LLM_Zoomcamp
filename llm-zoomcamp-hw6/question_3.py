import duckdb


DATABASE = 'agent_traces_pipeline.duckdb'


with duckdb.connect(DATABASE, read_only=True) as connection:
    columns = {
        row[0]
        for row in connection.execute(
            '''
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'agent_traces'
              AND table_name = 'records'
            '''
        ).fetchall()
    }

    token_columns = sorted(name for name in columns if 'input_token' in name)
    if not token_columns:
        raise RuntimeError('No input-token column found in agent_traces.records')

    print(f'token_columns={token_columns}')
    token_column = next(
        name
        for name in token_columns
        if name.endswith('gen_ai_usage_input_tokens')
    )
    result = connection.execute(
        f'''
        WITH candidate_traces AS (
            SELECT trace_id
            FROM agent_traces.records
            GROUP BY trace_id
            HAVING COUNT(*) = 5
               AND COUNT(*) FILTER (WHERE message = 'faq_agent run') = 1
            ORDER BY MAX(start_timestamp) DESC
            LIMIT 1
        )
        SELECT
            r.trace_id,
            SUM(COALESCE(r."{token_column}", 0)) AS input_tokens,
            COUNT(*) FILTER (WHERE r.span_name LIKE 'chat %') AS llm_calls
        FROM agent_traces.records AS r
        WHERE r.trace_id = (SELECT trace_id FROM candidate_traces)
          AND r.span_name LIKE 'chat %'
        GROUP BY r.trace_id
        '''
    ).fetchone()

if result is None:
    raise RuntimeError('Could not find the five-span Question 1 trace')

trace_id, input_tokens, llm_calls = result
print(f'trace_id={trace_id}')
print(f'llm_calls={llm_calls}')
print(f'input_tokens={input_tokens}')
