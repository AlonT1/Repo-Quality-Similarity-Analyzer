'''
A simple Python3 tool to detect similarities between files within a repository.

Document similarity code adapted from Jonathan Mugan's tutorial:
https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python
'''
import os
import tempfile
from collections import OrderedDict
import gensim
import pandas as pd
from nltk.tokenize import word_tokenize

source_code_file_extensions = [".c", ".cpp", ".cc", ".java", ".py", ".cs"]
repository_col_label = "Repository"
similarity_column_label = "Similarity (%)"
similarity_label_length = len(similarity_column_label)



def generate_matrix(repo_path, output_path):
    # get repository paths
    all_repos_root = repo_path
    repos_names = os.walk(all_repos_root).__next__()[1]  # get folders in "repositories" dir
    repositories_paths = [os.path.join(all_repos_root, repo_name) for repo_name in repos_names]

    df_main = pd.DataFrame(index=repositories_paths, columns=repositories_paths)
    df_main = df_main.fillna(0)

    # for each repo, get the paths of its files
    repo_source_files = OrderedDict()
    for repository_path in repositories_paths:
        repo_source_files[repository_path] = []
        for dirpath, _, filenames in os.walk(repository_path):
            for name in filenames:
                _, file_extension = os.path.splitext(name)
                if file_extension in source_code_file_extensions:
                    filename = os.path.join(dirpath, name)
                    repo_source_files[repository_path].append(filename)

    # parse content of all files per repo
    source_codes = OrderedDict()
    for repo_name, files in repo_source_files.items():
        for file in files:
            total_text = ""
            with open(file, encoding="ISO-8859-1") as f:
                total_text = total_text + total_text.join(f.read()) + " "
        source_codes[repo_name] = total_text

    # Create a similarity object of all the source code
    tokenized_files = []
    for source_code in source_codes.values():
        tokenized = [word.lower() for word in word_tokenize(source_code)]
        tokenized_files.append(tokenized)
    dictionary = gensim.corpora.Dictionary(tokenized_files)
    corpus_bow = [dictionary.doc2bow(tokenized_file) for tokenized_file in tokenized_files]
    tf_idf = gensim.models.TfidfModel(corpus_bow)
    sims = gensim.similarities.Similarity(tempfile.gettempdir() + os.sep, tf_idf[corpus_bow],
                                          num_features=len(dictionary))

    longest_repo_name = len(max(source_codes.keys(), key=len))  # calculated for spaces when printing

    # Create a similarity per file. To be compared with sims.
    all_similarity_averages = []
    for repo_name, source_code in source_codes.items():
        query_doc = [w.lower() for w in word_tokenize(source_code)]  # tokenize doc
        query_doc_bow = dictionary.doc2bow(query_doc)  # bag of words
        query_doc_tf_idf = tf_idf[query_doc_bow]  # doc as tf_idf vector

        # print title
        print("\n\n\n" + "Code duplication probability for " + repo_name )
        print("-" * (longest_repo_name + similarity_label_length))
        print(f"{repository_col_label.center(longest_repo_name)}"
              f" {similarity_column_label}")
        print("-" * (longest_repo_name + similarity_label_length))
        similarity_scores = []
        for similarity, source in zip(sims[query_doc_tf_idf], repositories_paths):
            if source == repo_name:
                df_main[repo_name].loc[source] = 100
                continue

            similarity_percentage = round((similarity * 100), 2)
            print(f"{source.ljust(longest_repo_name)} {similarity_percentage}")
            similarity_scores.append(similarity_percentage)
            df_main[repo_name].loc[source] = similarity_percentage
        avg_similarity = round(sum(similarity_scores) / len(similarity_scores), 2)
        print(f"Average similarity score for {repo_name} : {avg_similarity}")
        all_similarity_averages.append(avg_similarity)

    total_similarity_average = round(sum(all_similarity_averages) / len(all_similarity_averages), 2)
    print(f"\n\nTotal average similarity between all repositories:  {total_similarity_average}")

    # remove redundant file paths from file names - only the repository names will remain in the excel
    df_main.index = df_main.index.map(lambda x: x.rsplit("\\", 1)[-1])
    df_main.columns = df_main.columns.map(lambda x: x.rsplit("\\", 1)[-1])

    df_main.to_excel(output_path)
    print("The similarity Matrix was successfully generated.")


if __name__ == "__main__":
    generate_matrix()
