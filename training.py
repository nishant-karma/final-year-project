"""
Train ML Model to Classify / Identify the person using extracted face embeddings.
Additionally, calculate performance metrics like accuracy, confusion matrix, and classification report.
"""
from extract_embeddings import Extract_Embeddings
import cv2 
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import numpy as np

class Training:
    def __init__(self, embedding_path):
        self.embedding_path = embedding_path

    def load_embeddings_and_labels(self):
        """
        Load embeddings and labels from the pickle file.
        Returns:
            label: Label encoder instance.
            labels: Encoded labels as integers.
            embeddings: Face embeddings as numpy array.
            ids: Original face IDs.
        """
        data = pickle.loads(open(self.embedding_path, "rb").read())
        label = LabelEncoder()
        ids = np.array(data["face_ids"])
        labels = label.fit_transform(ids)
        embeddings = np.array(data["embeddings"])
        return label, labels, embeddings, ids

    def create_svm_model(self, labels, embeddings):
        """
        Train an SVM model on the provided embeddings and labels.
        Args:
            labels: Encoded labels as integers.
            embeddings: Face embeddings as numpy array.
        Returns:
            recognizer: Trained SVM model wrapped with probability calibration.
        """
        model_svc = LinearSVC(C=1.0, max_iter=5000)  # Increased max_iter to avoid convergence warnings
        recognizer = CalibratedClassifierCV(model_svc)
        recognizer.fit(embeddings, labels)
        return recognizer

    def calculate_metrics(self, embeddings, labels):
        """
        Split data, train the SVM, and calculate performance metrics.
        Args:
            embeddings: Face embeddings as numpy array.
            labels: Encoded labels as integers.
        """
        # Ensure data is shuffled
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, shuffle=True
        )

        # Train the model
        recognizer = self.create_svm_model(y_train, X_train)

        # Predict on the test set
        y_pred = recognizer.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Classification Report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Cross-Validation
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(recognizer, embeddings, labels, cv=skf)
        print(f"Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}%")

# Example usage
if __name__ == "__main__":
    trainer = Training(embedding_path="models/embeddings.pickle")
    label, labels, embeddings, ids = trainer.load_embeddings_and_labels()

    # Check for duplicate embeddings
    unique_embeddings = np.unique(embeddings, axis=0)
    print(f"Unique embeddings: {len(unique_embeddings)}, Total embeddings: {len(embeddings)}")

    trainer.calculate_metrics(embeddings, labels)
