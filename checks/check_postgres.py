import psycopg

from app.config import settings


def main() -> None:
    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user;")
            print(cur.fetchone())


if __name__ == "__main__":
    main()