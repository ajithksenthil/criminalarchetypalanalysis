import csv
import numpy as np
import analysis_integration as ai


def test_load_all_criminals_type1(tmp_path):
    csv_path = tmp_path / "Type1_Test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Age", "Life Event"])
        writer.writerow(["2000", "0", "Born"])
        writer.writerow(["2005", "5", "Started school"])
    data = ai.load_all_criminals_type1(tmp_path)
    assert "Test" in data
    assert data["Test"]["events"] == ["Born", "Started school"]
    assert len(data["Test"]["rows"]) == 2


def test_preprocess_text_basic():
    result = ai.preprocess_text("This is a TEST, 123! Running.")
    assert result == "test running"


def test_kmeans_cluster_basic():
    embeddings = np.array([[0, 0], [0, 1], [10, 10], [11, 11]])
    labels, model = ai.kmeans_cluster(embeddings, n_clusters=2, random_state=0)
    assert sorted(set(labels)) == [0, 1]
    assert len(labels) == 4

