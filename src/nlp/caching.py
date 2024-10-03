# from joblib import Memory


# Define the custom cache wrapper to set the module name manually
def cache(mem, module, **mem_kwargs):
    def cache_(f):
        f.__module__ = (
            module  # Set a custom module name to avoid multiple cache entries
        )
        f.__qualname__ = f.__name__  # Ensure the function name is consistent
        return mem.cache(f, **mem_kwargs)

    return cache_


# import os
# from joblib import Memory


# def dynamic_cache(notebook_name, cache_dir="../data/cache", **mem_kwargs):
#     """
#     Custom decorator to apply joblib caching with a dynamic cache directory.

#     Parameters:
#     - notebook_name (str): A unique name to identify the notebook (or environment).
#     - cache_dir (str): The base directory where cache files will be stored.
#     - **mem_kwargs: Additional keyword arguments to pass to joblib.Memory.
#     """

#     def decorator(func):
#         # Define the cache directory specific to the notebook
#         cache_path = os.path.join(cache_dir, notebook_name)
#         memory = Memory(cache_path, **mem_kwargs)

#         # Wrap the original function with joblib's cache functionality
#         cached_func = memory.cache(func)

#         def wrapper(*args, **kwargs):
#             # Call the cached function with the provided arguments
#             return cached_func(*args, **kwargs)

#         return wrapper

#     return decorator
