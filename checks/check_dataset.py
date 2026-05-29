from app.data_loader import load_customer_service_df


def main() -> None:
    df = load_customer_service_df()

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print()
    print("Categories:")
    print(df["category"].value_counts())
    print()
    print("Sample rows:")
    print(df[["instruction", "category", "intent", "response"]].head(3))


if __name__ == "__main__":
    main()