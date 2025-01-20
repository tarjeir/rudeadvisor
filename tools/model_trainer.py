import tensorflow as tf


from tensorflow._api.v2.sparse import expand_dims
import typer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List
from dataclasses import dataclass
import json
import os
from pathlib import Path
import keras

app = typer.Typer()


@dataclass
class ItemFeatures:
    item_id: str
    domain: str
    url: str
    topic_words: List[str]
    label: int


@dataclass
class ItemFeaturesWithScore:
    item_features: ItemFeatures
    score: float


@dataclass
class ItemFeaturesWithTensor:
    item_id: str
    item_features: ItemFeatures
    tensor: tf.Tensor


@dataclass
class UserFeatures:
    user_id: str
    user_topics: list[str]


@dataclass
class UserFeaturesWithTensor:
    user_features: UserFeatures
    tensor: tf.Tensor


class TwoTowerModel(keras.Model):
    def __init__(
        self,
        user_features_input_shape: tuple,
        items_feautures_input_shape: tuple,
        embedding_dim=32,
    ):
        super().__init__()

        # User tower
        self.user_model = keras.Sequential(
            [
                keras.layers.Input(shape=user_features_input_shape),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(embedding_dim),
            ]
        )
        # Item tower
        self.item_model = keras.Sequential(
            [
                keras.layers.Input(shape=items_feautures_input_shape),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(embedding_dim),
            ]
        )

    def get_recommendation(
        self,
        user_features_tensor: UserFeaturesWithTensor,
        item_features_list: list[ItemFeaturesWithTensor],
        top_k: int = 5,
    ):

        user_embedding = self.user_model(
            tf.expand_dims(user_features_tensor.tensor, axis=0)
        )
        user_embedding = tf.reshape(user_embedding, [-1, user_embedding.shape[-1]])

        items_with_scores = []

        for item in item_features_list:
            item_embedding = self.item_model(tf.expand_dims(item.tensor, axis=0))
            item_embedding = tf.reshape(item_embedding, [-1, item_embedding.shape[-1]])
            # Compute similarity
            similarity = tf.reduce_sum(user_embedding * item_embedding, axis=1)
            probability = tf.sigmoid(similarity)
            items_with_scores.append((item.item_features, probability))

        # Sort items by similarity score in descending order
        items_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top K recommended items
        return [
            ItemFeaturesWithScore(item, score)
            for item, score in items_with_scores[:top_k]
        ]

    def call(self, inputs):
        features = inputs
        user_features = features["user_features"]
        item_features = features["item_features"]

        user_embeddings = self.user_model(user_features)
        item_embeddings = self.item_model(item_features)

        # Flatten the last two dimensions and compute dot product
        user_embeddings = tf.reshape(
            user_embeddings, [-1, user_embeddings.shape[-1]]
        )  # Shape: (None, embedding_dim)
        item_embeddings = tf.reshape(
            item_embeddings, [-1, item_embeddings.shape[-1]]
        )  # Shape: (None, embedding_dim)

        # Compute dot product to get a scalar logit for each pair
        similarity = tf.reduce_sum(
            user_embeddings * item_embeddings, axis=1
        )  # Shape: (None,)

        # Apply the sigmoid to produce a probability
        probability = tf.sigmoid(similarity)  # Shape: (None,)

        return probability


def read_json_files(directory: Path, label: int) -> List[ItemFeatures]:
    item_features_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    item_id = data["metadata"]["hash"]
                    domain = data["metadata"]["domain"]
                    url = data["metadata"]["url"]

                    bert_features = data["features"]["bert_features"]["topics"]
                    if len(bert_features) > 0:
                        topic_words = bert_features[0]["words"]
                    else:
                        lda_features = data["features"]["lda_features"]["topics"]
                        topic_words = lda_features[0]["words"]

                    item_features = ItemFeatures(
                        item_id, domain, url, topic_words, label
                    )
                    item_features_list.append(item_features)
    return item_features_list


from sklearn.preprocessing import LabelEncoder


def create_label_encoder_for_items(preferences: List[ItemFeatures]) -> LabelEncoder:
    flat_preferences: list[str] = [
        topic_word
        for preference in preferences
        for topic_word in preference.topic_words
    ]
    encoder = LabelEncoder()
    encoder.fit(flat_preferences)
    return encoder


def create_label_encoder_for_users(preferences: List[UserFeatures]) -> LabelEncoder:
    flat_topic: list[str] = [
        user_topic
        for preference in preferences
        for user_topic in preference.user_topics
    ]
    encoder = LabelEncoder()
    encoder.fit(flat_topic)
    return encoder


def encode_preferences(
    label_encoder: LabelEncoder,
    preferences: List[ItemFeatures],
    max_length_of_tensor: int | None = None,
) -> tuple[list[ItemFeaturesWithTensor], int | None]:

    encoded_preferences = [
        label_encoder.transform(pref.topic_words).tolist() for pref in preferences
    ]

    if max_length_of_tensor:
        max_length = max_length_of_tensor
    else:
        max_length = max(len(encoded) for encoded in encoded_preferences)

    encoded_preferences_per_item = []

    for encoded, pref in zip(encoded_preferences, preferences):
        padded_encoded = keras.preprocessing.sequence.pad_sequences(
            [encoded], maxlen=max_length, padding="post"
        )

        encoded_preferences_per_item.append(
            ItemFeaturesWithTensor(
                item_id=pref.item_id,
                item_features=pref,
                tensor=tf.convert_to_tensor(padded_encoded, dtype=tf.float32),
            )
        )

    return (encoded_preferences_per_item, max_length)


def encode_users(
    label_encoder: LabelEncoder,
    preferences: List[UserFeatures],
    max_length_of_tensor: int | None = None,
) -> tuple[list[UserFeaturesWithTensor], int | None]:
    encoded_preferences = [
        label_encoder.transform(pref.user_topics).tolist() for pref in preferences
    ]
    if max_length_of_tensor:
        max_length = max_length_of_tensor
    else:
        max_length = max(len(encoded) for encoded in encoded_preferences)

    encoded_preferences_per_item = []

    for encoded, pref in zip(encoded_preferences, preferences):
        padded_encoded = keras.preprocessing.sequence.pad_sequences(
            [encoded], maxlen=max_length, padding="post"
        )

        encoded_preferences_per_item.append(
            UserFeaturesWithTensor(
                user_features=pref,
                tensor=tf.convert_to_tensor(padded_encoded, dtype=tf.float32),
            )
        )

    return (encoded_preferences_per_item, max_length)


def create_dataset_from_a_user_features_and_items(
    item_features: list[ItemFeaturesWithTensor],
    user_features: list[UserFeaturesWithTensor],
    batch_size: int = 85,
):
    features = {"item_features": [], "user_features": []}
    labels = []

    for item_feature, user_feature in zip(item_features, user_features):
        features["item_features"].append(item_feature.tensor)
        features["user_features"].append(user_feature.tensor)
        labels.append(
            tf.constant(float(item_feature.item_features.label), dtype=tf.float32)
        )

    return (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .shuffle(len(item_features))
        .batch(batch_size)
    )


def create_train_and_validation_dataset(
    item_features: list[ItemFeaturesWithTensor],
    user_features: list[UserFeaturesWithTensor],
    batch_size: int = 85,
):

    train_items, validation_items = train_test_split(
        item_features, test_size=0.2, random_state=42
    )

    train_dataset = create_dataset_from_a_user_features_and_items(
        train_items, user_features, batch_size
    )
    val_dataset = create_dataset_from_a_user_features_and_items(
        validation_items, user_features, batch_size
    )

    return (train_dataset, val_dataset)


def inspect_dataset(dataset, name="Dataset"):
    print(f"\nInspecting {name}:")
    for batch in dataset.take(1):
        print("\nBatch structure:")
        if isinstance(batch, dict):
            for key, value in batch.items():
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"  First few values: {value.numpy()[:5]}")
        else:
            print(f"Batch type: {type(batch)}")
            print(f"Batch elements: {len(batch)}")
            for i, element in enumerate(batch):
                print(f"\nElement {i}:")
                if isinstance(element, dict):
                    for k, v in element.items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  shape={element.shape}, dtype={element.dtype}")


def compile_model(
    user_features_shape: tuple, items_features_shape: tuple, embedding_dim: int
) -> TwoTowerModel:
    model = TwoTowerModel(
        user_features_input_shape=user_features_shape,
        items_feautures_input_shape=items_features_shape,
        embedding_dim=embedding_dim,
    )
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.Precision(), keras.metrics.Recall()],
    )
    return model


@app.command()
def main(
    positive_labels_path: Path = typer.Option(
        ..., help="Path to positive labels directory"
    ),
    negative_labels_path: Path = typer.Option(
        ..., help="Path to negative labels directory"
    ),
    unclassified_path: Path = typer.Option(..., help="Path to unclassified"),
):
    """Create a numeric dataset for a Keras model."""
    positive_items = read_json_files(positive_labels_path, 1)
    negative_items = read_json_files(negative_labels_path, 0)
    unclassified_items = read_json_files(unclassified_path, 0)

    label_encoder = create_label_encoder_for_items(
        positive_items + negative_items + unclassified_items
    )

    encoded_items, max_length_for_item_tensor = encode_preferences(
        label_encoder, positive_items + negative_items
    )
    shapes = set()
    for item in encoded_items:
        if len(shapes) > 1:
            raise ValueError(
                f"We found a more than one shape in the list of items {shapes}"
            )
        shapes.add(f"{item.tensor.shape[0]}, {item.tensor.shape[1]}")

    user_features = [
        UserFeatures(
            user_id="1",
            user_topics=["ai", "machine learning", "llm", "data science", "data"],
        )
    ]
    label_encode_for_user_features = create_label_encoder_for_users(user_features)

    user_features_encoded, max_length_for_user_tensor = encode_users(
        label_encode_for_user_features, user_features
    )

    train_dataset, validation_dataset = create_train_and_validation_dataset(
        item_features=encoded_items, user_features=user_features_encoded
    )

    inspect_dataset(dataset=train_dataset)
    inspect_dataset(dataset=validation_dataset)
    model = compile_model(
        user_features_shape=(1, 5), items_features_shape=(1, 10), embedding_dim=64
    )

    model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

    unclassified_items_encoded, _ = encode_preferences(
        label_encoder, unclassified_items, max_length_for_item_tensor
    )
    user_features, _ = encode_users(
        label_encode_for_user_features,
        [UserFeatures(user_id="2", user_topics=["data"])],
        max_length_for_user_tensor,
    )
    print(user_features)
    print(max_length_for_user_tensor)
    print(
        model.get_recommendation(
            user_features_tensor=user_features[0],
            item_features_list=unclassified_items_encoded,
            top_k=10,
        )
    )


if __name__ == "__main__":
    app()
