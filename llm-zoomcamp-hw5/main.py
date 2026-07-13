import sys

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from sqlite_span_exporter import SQLiteSpanExporter

sys.stdout.reconfigure(encoding="utf-8")

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)

# Configure OpenTelemetry before importing application code.
from starter import rag


def main():
    query = "How does the agentic loop keep calling the model until it stops?"

    answer = rag.rag(query)
    print(answer)


if __name__ == "__main__":
    main()
