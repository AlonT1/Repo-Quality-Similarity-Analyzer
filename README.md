# Repo-Quality-Similarity-Analyzer

This tool serves two purposes:
Given a collection of git repositories stored locally in a folder:
1. Generate an excel file that contains a similarity score between each pair of repositories based on the contents of their files.
For example:
![similarity scores](https://i.ibb.co/T2kwGcq/similarity-scores.png)

2. Give a quality score to the repositories based on the following formula:
![formula](https://i.ibb.co/Lkd1HfC/formula.jpg)
where maxStars, maxForks,... are taken from the repository with the highest quality in the collection.

# User interface
![analyzer](https://i.ibb.co/W2CGH6v/program.jpg)
