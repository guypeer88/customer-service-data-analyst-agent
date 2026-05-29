from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from app.config import settings


def main() -> None:
    pool = ConnectionPool(
        conninfo=settings.database_url,
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )

    checkpointer = PostgresSaver(pool)

    print("Running checkpointer setup...")
    checkpointer.setup()

    print("Postgres checkpointer is ready.")


if __name__ == "__main__":
    main()