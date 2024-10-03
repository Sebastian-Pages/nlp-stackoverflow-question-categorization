import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample data for testing
data = {
    "CleanPost": [
        "This is a sample post about AI.",
        "Another post discussing machine learning.",
        "Deep learning is a subset of machine learning.",
        "This is a sample post about AI and ML.",
    ],
    "Tags": [
        ["AI", "technology"],
        ["machine learning", "AI"],
        ["deep learning"],
        ["AI", "machine learning"],
    ],
}

# Create DataFrame
df = pd.DataFrame(data)

# Split into train and test datasets
train_df, test_df = train_test_split(
    df[["CleanPost", "Tags"]], test_size=0.2, random_state=42
)

# Expected sizes for train and test datasets
expected_train_size = int(len(df) * 0.8)  # 80% for training
expected_test_size = len(df) - expected_train_size  # 20% for testing


# Jaccard Score Function
def jaccard_score(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    if union == 0:
        return 0
    return intersection / union


# Test for train-test split lengths
def test_train_test_split_length():
    assert (
        len(train_df) == expected_train_size
    ), f"Expected train size {expected_train_size}, but got {len(train_df)}."
    assert (
        len(test_df) == expected_test_size
    ), f"Expected test size {expected_test_size}, but got {len(test_df)}."


# Test for Jaccard score calculation
def test_jaccard_score():
    true_tags = ["tag1", "tag2", "tag3"]
    predicted_tags_1 = ["tag1", "tag2"]  # 2 correct tags
    predicted_tags_2 = ["tag4", "tag5"]  # No correct tags
    predicted_tags_3 = []  # No tags

    # Expected Jaccard scores
    expected_score_1 = 2 / 3  # 2 correct out of 3 unique tags
    expected_score_2 = 0  # No correct tags
    expected_score_3 = 0  # No tags at all

    # Calculate Jaccard scores
    score_1 = jaccard_score(true_tags, predicted_tags_1)
    score_2 = jaccard_score(true_tags, predicted_tags_2)
    score_3 = jaccard_score(true_tags, predicted_tags_3)

    # Assertions
    assert np.isclose(
        score_1, expected_score_1
    ), f"Expected {expected_score_1}, but got {score_1}."
    assert np.isclose(
        score_2, expected_score_2
    ), f"Expected {expected_score_2}, but got {score_2}."
    assert np.isclose(
        score_3, expected_score_3
    ), f"Expected {expected_score_3}, but got {score_3}."
