import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import sys
import os
from joblib import Memory
from wordcloud import WordCloud
from collections import Counter


# Create a Memory object for caching
memory = Memory("../data/cache", verbose=0)


@memory.cache(ignore=["df"])
def count_top_tags(df, count_threshold):
    tag_counts = Counter(tag for tags in df["Tags"] for tag in tags)
    tags_above_threshold = sum(
        1 for tag, count in tag_counts.items() if count > count_threshold
    )
    return tags_above_threshold


@memory.cache(ignore=["df"])
def plot_tags_count(
    df,
    column_name="Tags",
    title="Tag Distribution",
    figsize=(10, 3),
    color="#7851A9",  # Royal Purple color
    xlabel="Tags",
    ylabel="Counts",
):
    # Count the occurrences of each tag
    tag_counts = Counter(tag for tags in df[column_name] for tag in tags)

    # Convert the tag counts to a DataFrame
    tag_counts_df = pd.DataFrame(tag_counts.items(), columns=["Tag", "Count"])

    # Sort the DataFrame by Count
    tag_counts_df = tag_counts_df.sort_values(by="Count", ascending=False)

    # Create the line plot (no markers)
    plt.figure(figsize=figsize)
    sns.lineplot(x="Tag", y="Count", data=tag_counts_df, color=color)

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove x-axis labels (ticks)
    plt.xticks([], [])

    return plt.gcf()


# @memory.cache(ignore=["df"])
def plot_word_counts(
    df,
    column_name,
    title="Word Counts",
    top_n=10,
    figsize=(10, 10),
    color="#7851A9",  # Royal Purple color
    xlabel="Words",
    ylabel="Counts",
):
    # Combine all text in the specified column and drop any NA values
    body_text = df[column_name].dropna().str.cat(sep=" ")

    # Tokenize the text
    tokens = word_tokenize(body_text)

    # Create frequency distribution of tokens
    freq_dist = FreqDist(tokens)

    # Convert FreqDist to DataFrame for plotting
    df_word_counts = pd.DataFrame(
        freq_dist.most_common(top_n), columns=["Word", "Count"]
    )

    # Ensure the data types are correct
    df_word_counts["Word"] = df_word_counts["Word"].astype(str)
    df_word_counts["Count"] = df_word_counts["Count"].astype(int)

    print(df_word_counts.dtypes)  # Debug: check the types before plotting

    # Create the plot
    plt.figure(figsize=figsize)
    sns.barplot(x="Count", y="Word", data=df_word_counts, color=color)

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return plt.gcf()


# @memory.cache(ignore=["df"])
def generate_word_cloud(
    df, column_name, title="Word Cloud", width=800, height=400, background_color="white"
):
    # Combine all text in the specified column and drop any NA values
    body_text = df[column_name].dropna().str.cat(sep=" ")

    # Tokenize the text
    tokens = word_tokenize(body_text)

    # Create frequency distribution of tokens
    freq_dist = FreqDist(tokens)

    # Convert FreqDist to a dictionary for word cloud
    word_freq_dict = dict(freq_dist)

    # Generate the word cloud
    wordcloud = WordCloud(
        width=width, height=height, background_color=background_color
    ).generate_from_frequencies(word_freq_dict)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")

    # Add title with royal purple color
    plt.title(title, fontsize=16, color="#7851A9")  # Royal Purple color
    plt.axis("off")  # Hide the axis

    return plt.gcf()


@memory.cache(ignore=["df"])
def plot_posts_by_time(
    df,
    date_column,
    time_unit="year",  # can be 'year', 'month', 'day'
    title="Number of Posts by Time Period",
    xlabel="Time Period",
    ylabel="Number of Posts",
    color="#7851A9",  # Royal Purple
    figsize=(10, 3),
    rotation=45,
):

    # Ensure the date_column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Group by the specified time unit
    if time_unit == "year":
        df["TimePeriod"] = df[date_column].dt.year
    elif time_unit == "month":
        df["TimePeriod"] = df[date_column].dt.to_period("M")  # 'M' for monthly grouping
    elif time_unit == "day":
        df["TimePeriod"] = df[date_column].dt.to_period("D")  # 'D' for daily grouping
    else:
        raise ValueError("time_unit must be 'year', 'month', or 'day'.")

    # Count the number of posts per time period
    post_counts = df.groupby("TimePeriod").size().reset_index(name="PostCount")

    # Plot the data using sns.catplot with a single color
    g = sns.catplot(
        data=post_counts,
        x="TimePeriod",
        y="PostCount",
        kind="bar",
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        color=color,  # Use a single color for all bars
        legend=False,  # Disable the legend
    )

    # Add titles and labels
    g.figure.suptitle(title, y=1.05)  # Adjust title position
    g.set_axis_labels(xlabel, ylabel)
    g.set_xticklabels(rotation=rotation)

    return g.figure


# @cache(mem, "visualization")
@memory.cache(ignore=["df"])
def pretty_sample(df, columns, sorting_col, n):

    df = df.sort_values(by=sorting_col, ascending=False)[columns]

    # Get a mix
    top_n = df.head(n)
    mid_index = len(df) // 2
    mid_n = df.iloc[mid_index - n // 2 : mid_index + n // 2 + (n % 2)]
    bottom_n = df.tail(n)
    result = pd.concat([top_n, mid_n, bottom_n], ignore_index=True)

    # Style with different colors for each section
    styled_result = result.style.apply(
        lambda x: [
            (
                "background-color: rgba(0, 255, 0, 0.07)"
                if i < n
                else (
                    "background-color: rgba(255, 255, 0, 0.07)"
                    if n <= i < n + len(mid_n)
                    else "background-color: rgba(255, 0, 0, 0.07)"
                )
            )
            for i in range(len(x))
        ],
        axis=0,
    )

    return styled_result
