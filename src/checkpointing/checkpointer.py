import os
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver


def get_checkpointer() -> PostgresSaver:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer
