import duckdb


with duckdb.connect('agent_traces_pipeline.duckdb', read_only=True) as connection:
    tables = connection.execute(
        '''
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'agent_traces'
        ORDER BY table_name
        '''
    ).fetchall()

print(f'table_count={len(tables)}')
for (table_name,) in tables:
    print(table_name)
