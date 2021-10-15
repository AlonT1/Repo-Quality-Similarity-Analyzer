import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style
from scipy import stats
import matplotlib.ticker as ticker
style.use('ggplot')

BINS_AMOUNT = 10


# counts "issues" from the "issues" csv and merges the count as a new column into the main repo_csv file
def merge_issues(repo_csv, issues_csv, output_path):
    # load the main repository csv
    df_repos = pd.read_csv(repo_csv)

    # Get issues from the issues csv
    df_issues = pd.read_csv(issues_csv)
    series = df_issues["repo_id"].value_counts()
    df_issue_counter = pd.DataFrame({"repo_id": series.index, "issues": series.values})

    # merge the issues to together with the main repository csv.
    # repos which don't exist in the issues csv are filled with 0 issues
    df_repos_merged = df_repos.merge(df_issue_counter, on="repo_id", how="left").fillna(0, downcast='infer')

    # write csv
    df_repos_merged.to_excel(output_path, index=False)


# calculates the similarity score based on the similarity matrix
def similarity_metric(values, bins_amount):
    bin_size = 100 / bins_amount
    values = values.drop(values.name)  # dropping the 100 from the project itself
    bins = np.arange(0, 100 + bin_size, bin_size)
    weights = np.arange(bin_size / 2, 100 + bin_size / 2, bin_size)
    statistic, bin_edges, binnumber = stats.binned_statistic(values, values, 'count', bins)
    metric = sum(statistic * weights) / len(values)
    return round(metric, 5)


def calculate_similarity_scores(path, domain_name, df_similarity, bins):
    df_similarity["score"] = df_similarity.apply(similarity_metric, axis=1, bins_amount=bins)
    df_similarity.to_excel(f"{path}\\{domain_name} - similiarity matrix scored.xlsx")
    return df_similarity.sort_values(by="score", ascending=False)


# forms a box plot and retrieves the *maximum value from a dataframe column
# we define maximum as the value at the top whisker in the box plot (IQR+Q3*1.5)
def max_metrics(df, columns):
    values = []
    for col in columns:
        dic = df.boxplot(column=col, return_type="dict", showfliers=False, whis=1.5)
        top_whisker = dic["caps"][1].get_ydata()[0]
        ninth_quantile = df.quantile(0.9)[col]
        max_value = top_whisker if (top_whisker != 0) else ninth_quantile
        values.append(max_value)
    return values


# calculates the quality score based on specific parameters in the quality csv
def calculate_quality_score(path, domain_name, df):
    # calculate score
    columns = ["forks_count", "stargazers_count", "subscribers_count", "issues"]
    maxForks, maxStargazers, maxWatchers, maxIssues = max_metrics(df, columns)
    df["score"] = df.apply(lambda row: row["stargazers_count"] * 0.65 / maxStargazers +
                                       row["forks_count"] * 0.2 / maxForks + row[
                                           "watchers_count"] * 0.05 / maxWatchers +
                                       row["issues"] * 0.05 / maxIssues, axis=1)

    # we clip the final score at the 90% quantile - all scores above equal 100
    maxScore = df.quantile(0.9)["score"]
    df["score"] = np.clip(df["score"], a_min=df["score"].min(), a_max=maxScore)
    df["score"] = df["score"].round(3)

    #  normalize the score
    maxQuality = df["score"].max()
    minQuality = df["score"].min()
    df["score"] = df.apply(lambda row: (row["score"] - minQuality) / (maxQuality - minQuality) * 100, axis=1)

    df["score"] = df["score"].round(5)
    df.to_excel(f"{path}\\{domain_name} - repository quality table scored.xlsx", index=False)
    return df.sort_values(by="score", ascending=False)


def combine_quality_similarity(path, domain_name, df_similarity, df_quality):
    # treating df_similarity
    df_similarity_scores = df_similarity["score"].to_frame("similarity").reset_index()
    df_similarity_scores = df_similarity_scores.rename(columns={"index": "repo_id"})

    # treating df_quality_scores
    df_quality_scores = df_quality[["repo_id", "score"]]
    df_quality_scores = df_quality_scores.rename(columns={"score": "quality"})

    df_combined = pd.merge(df_similarity_scores, df_quality_scores, how="inner", on="repo_id")
    df_combined = df_combined.set_index("repo_id")
    df_combined.to_excel(f"{path}\\{domain_name} - quality vs similarity.xlsx")
    return df_combined


def plot_correlation(path, domain_name, df, title):
    sns.lmplot(x='similarity', y='quality', data=df, fit_reg=True, ci=None,
               aspect=1.8, line_kws={'color': 'blue', 'alpha': 0.6})
    plt.xlabel("similarity", fontsize=12)
    plt.ylabel("quality", fontsize=12)
    plt.title(title, fontsize=15)
    df.corr().to_excel(f"{path}\\{domain_name} - correleation score.xlsx")
    plt.savefig(f'{path}\\{title}.png', bbox_inches='tight')


def histogram(path, scores, title, face_color, text_offset, bins_amount=None):
    # plotting histogram
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_ylabel('Amount of repositories', size=14)
    ax.set_xlabel('Scores', size=14)
    n, bins, _ = ax.hist(scores, bins=bins_amount, facecolor=face_color, alpha=0.5,
                         edgecolor='black', linewidth=2)
    ax.set_xticks(np.round(bins, 2))

    ax.set_title(title, size=15)
    ax.tick_params(axis='both', which='major', labelsize=14)

    for index, value in zip(bins, n):
        ax.text(x=index + text_offset, y=value, s=f"{int(value):,}",
                color='black', fontweight='bold', size=15)
    plt.savefig(f'{path}\\{title}.png', bbox_inches='tight')


def metric_boxplots(df, path, title):
    columns = ["forks_count", "stargazers_count", "subscribers_count", "issues"]
    metrics = []
    for column in columns:
        metrics.append(df[column].values)
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_yscale('symlog')
    box_stats = ax.boxplot(metrics, labels=["forks_count", "stargazers_count", "watchers_count", "issues"],
                           autorange=False, whis=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x)))
    # ax.yaxis.set_major_formatter(FixedFormatter([0,10,50,100,200,300,400,500]))

    boxes = [whiskers.get_ydata()[0] for whiskers in box_stats["whiskers"]]  # where 1st whisker ends & 2nd starts
    whiskers = [whiskers.get_ydata()[1] for whiskers in box_stats["caps"]]
    extreme_outliers = [np.sort(outlier.get_ydata())[-1] if
                        len(outlier.get_ydata()) != 0 else 0 for outlier in box_stats["fliers"]]

    # create indexes for text creation
    indexes = np.arange(len(whiskers))
    indexes = np.repeat(indexes + 1, 2)  # repeat index to enable the drawing of all data on each boxplot

    ax.tick_params(axis='both', which='major', labelsize=13)
    # plot numbers on boxes
    for index, value in zip(indexes, boxes):
        ax.text(x=index - 0.1, y=value, s=f"{int(value):}", color='blue', fontweight='bold', size=12);
    # plot numbers on boxes
    for index, value in list(zip(indexes, whiskers)):
        ax.text(x=index - 0.1, y=value, s=f"{int(value):}", color='blue', size=12);
    # plot numbers on outliers
    for index, value in list(zip(np.arange(len(extreme_outliers)) + 1, extreme_outliers)):
        if value != 0:
            ax.text(x=index - 0.1, y=value, s=f"{int(value):}", color='blue', size=12);

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("metrics")
    ax.set_ylabel("values (log scale)");
    plt.savefig(f'{path}\\{title}.png', bbox_inches='tight')


def show_insignificants(df, dataset_name):
    df_insignificants = pd.DataFrame({"All row values are less than or equal to": [], "rows": []}, index=None,
                                     dtype=int)
    columns = ["forks_count", "stargazers_count", "subscribers_count", "issues"]
    for value in range(11):
        num_rows = len(df.loc[(df[columns] <= value).all(axis=1)][columns])
        df_insignificants.loc[len(df_insignificants)] = [value, num_rows]
    print(f'Total rows in the {dataset_name} dataset: {len(df)}')
    return df_insignificants


def similarity_scatter(df_similarity, path, title):
    fig, ax = plt.subplots(figsize=(16, 9))
    iteration = 0
    df = df_similarity.sort_values(by="score").iloc[0:, :-1]
    for row in df.index:
        repo = df.loc[row].drop(df.loc[row].name)  # row as a series
        index = np.full(shape=len(repo.values), fill_value=iteration, dtype=int)
        iteration = iteration + 1
        ax.scatter(index, repo.values, s=20)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10.00))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10.00))

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(title)
    ax.set_xlabel(f'repositories - Numbered from 0 to {len(df)}')
    ax.set_ylabel('scores')
    plt.savefig(f'{path}\\{title}.png', bbox_inches='tight')


def main(similarity_matrix, repositories_table, domain_name, path):
    df_similarity = pd.read_excel(similarity_matrix, sheet_name=0, index_col=0)
    df_similarity = calculate_similarity_scores(path, domain_name, df_similarity,
                                                bins=BINS_AMOUNT)  # calculate similarity
    similarity_scatter(df_similarity, path, title=f"{domain_name} - Similarity Scores Scatter")
    histogram(path, df_similarity["score"], title=f"{domain_name} - Similarity Scores Distribution",
              face_color='red', text_offset=0.1, bins_amount=BINS_AMOUNT)
    df_quality = pd.read_excel(repositories_table)
    df_quality = calculate_quality_score(path, domain_name, df_quality)
    metric_boxplots(df_quality, path, title=f"{domain_name} - Quality Dataset Metrics Analysis")
    histogram(path, df_quality["score"], title=f"{domain_name} - Quality Scores Distribution",
              face_color='blue', text_offset=2.5)
    df_combined = combine_quality_similarity(path, domain_name, df_similarity, df_quality)
    plot_correlation(path, domain_name, df_combined, title=f"{domain_name} - Quality vs Similarity Correlation")
    print("metrics generated successfully")
