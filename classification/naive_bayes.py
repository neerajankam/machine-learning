import kaggle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NaiveBayes:
    """
    A class representing the Naive Bayes classifier for fraud detection.

    This class provides methods to download a credit card fraud dataset from Kaggle,
    authenticate with the Kaggle API, and perform prediction using the Naive Bayes algorithm.
    """

    dataset_path: str = None

    @classmethod
    def run(cls) -> None:
        """Runs the Naive Bayes classifier."""
        cls.authenticate()
        cls.download_dataset()
        cls.predict()

    @classmethod
    def download_dataset(cls) -> str:
        """Downloads the dataset if not already downloaded and returns the dataset path."""
        if not cls.dataset_path:
            kaggle.api.dataset_download_files(
                "mlg-ulb/creditcardfraud", path="data/", unzip=True
            )
            cls.dataset_path = "data/creditcard.csv"
        return cls.dataset_path

    @classmethod
    def authenticate(cls) -> None:
        """Authenticates the Kaggle API."""
        kaggle.api.authenticate()

    @classmethod
    def predict(cls) -> None:
        """Performs prediction using Naive Bayes classifier."""
        # Read the dataset into a pandas dataframe
        data = pd.read_csv("data/creditcard.csv")

        # Load the features into X and the target into Y
        X = data.drop("Class", axis=1)
        y = data["Class"]

        # Split the dataset into training and testing datasets. Use 80% for training and 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Load the model and train it on the data
        gaussian_naive_bayes = GaussianNB()
        gaussian_naive_bayes.fit(X_train, y_train)

        # Predict the output
        y_pred = gaussian_naive_bayes.predict(X_test)

        # Print the metrics
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)


if __name__ == "__main__":
    NaiveBayes().run()
