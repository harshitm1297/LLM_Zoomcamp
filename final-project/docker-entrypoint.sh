#!/bin/sh
set -eu

if [ "$(id -u)" = "0" ] && [ "${CONTAINER_RUN_AS_ROOT:-false}" != "true" ]; then
    mkdir -p /app/data /app/chroma_db /app/.cache/huggingface
    chown -R appuser:appuser /app/data /app/chroma_db /app/.cache/huggingface
    chmod -R u+rwX /app/data /app/chroma_db /app/.cache/huggingface
    export HOME=/home/appuser
    exec setpriv --reuid=10001 --regid=10001 --init-groups "$@"
fi

exec "$@"
