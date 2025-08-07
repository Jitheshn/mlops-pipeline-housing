from sklearn.datasets import fetch_california_housing


def load_and_save_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv("data/raw/housing.csv", index=False)


if __name__ == "__main__":
    load_and_save_data()
