import os
import pickle


def check_pickle_cache(method_name, prefix="../data/embeddings"):

    # Generate a filename for the cache
    cache_id = os.path.join(prefix, f"{method_name}.pkl")

    # Check if the pickle file exists
    if os.path.exists(cache_id):
        print(f"Loading cached data from {cache_id}")
        with open(cache_id, "rb") as f:
            return pickle.load(f)

    # Return None if the cache does not exist
    print(f"No cache found for {method_name}.")
    return None


def save_pickle_cache(data, identifier, prefix="../data/embeddings"):

    # Create cache directory if it doesn't exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Generate a filename for the cache
    cache_id = os.path.join(prefix, f"{identifier}.pkl")

    # Save data to pickle file
    print(f"Saving new data to {cache_id}")
    with open(cache_id, "wb") as f:
        pickle.dump(data, f)


# def check_model_cache(model_name, embedding_name, prefix="../data/cache/"):
#     # Generate a filename for the cache
#     cache_id = os.path.join(prefix, f"{model_name}_{embedding_name}.pkl")

#     # Check if the pickle file exists
#     if os.path.exists(cache_id):
#         print(f"Loading cached model from {cache_id}")
#         with open(cache_id, "rb") as f:
#             return pickle.load(f)

#     # Return None if the cache does not exist
#     print(f"No cache found for {model_name} with {embedding_name}.")
#     return None


# def save_model_cache(model, model_name, embedding_name, prefix="../data/cache/"):
#     # Create cache directory if it doesn't exist
#     if not os.path.exists(prefix):
#         os.makedirs(prefix)

#     # Generate a filename for the cache
#     cache_id = os.path.join(prefix, f"{model_name}_{embedding_name}.pkl")

#     # Save the model to a pickle file
#     print(f"Saving model to {cache_id}")
#     with open(cache_id, "wb") as f:
#         pickle.dump(model, f)
