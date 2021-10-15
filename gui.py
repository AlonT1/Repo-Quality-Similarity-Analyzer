import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import matrix_generator as mg
import metrics_functions as qf


def check_paths(not_empty, exists, extensions):
    paths = [ent_repos_path.get(), ent_output_path.get(), ent_repocsv_path.get(), ent_issuescsv_path.get(),
             ent_mergedxlsx_path.get(), ent_domainame_path.get(), ent_similaritytable_path.get(),
             ent_repotable_path.get(), ent_metrics_path.get()]
    labels = [lbl_repos['text'], lbl_output['text'], lbl_repocsv['text'],
              lbl_issuescsv['text'], lbl_mergedxlsx['text'], lbl_domainame['text'], lbl_similaritytable['text'],
              lbl_repotable['text'], lbl_metrics['text']]

    # checking paths are not empty
    for i in not_empty:
        if len(paths[i]) == 0:
            tk.messagebox.showwarning("showwarning", labels[i] + " path is empty")
            return False
    # checking paths exist
    for i in exists:
        if not os.path.exists(paths[i]):
            tk.messagebox.showwarning("showwarning", labels[i] + " path doesn't exist")
            return False
    # checking extensions
    for i in extensions:
        if paths[i].find(extensions[i], -5) == -1:
            print(paths[i], extensions[i])
            tk.messagebox.showwarning("showwarning", labels[i] + " path is not an " + extensions[i] + " file.")
            return False

    return True


def choose_directory(path):
    folder_selected = filedialog.askdirectory()
    path.delete(0, tk.END)
    path.insert(0, folder_selected)


def open_file(path):
    filename = filedialog.askopenfilename()
    if len(filename) != 0:
        path.delete(0, tk.END)
        path.insert(0, filename)


def save_file(path, type):
    if type == "xlsx":
        defaultextension = "xlsx"
        filetypes = [("Excel Workbook", "*.xlsx")]
    filepath = filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
    if len(filepath) != 0:
        path.delete(0, tk.END)
        path.insert(0, filepath)

# step 1
def generate_similarity_matrix():
    if not check_paths(not_empty=[0, 1], exists=[0], extensions={1: "xlsx"}):
        return False
    repo_path = ent_repos_path.get()
    output_path = ent_output_path.get()
    mg.generate_matrix(repo_path, output_path)


# step 2
def generate_merged_xlsx():
    if not check_paths(not_empty=[2, 3, 4], exists=[2, 3], extensions={2: "csv", 3: "csv", 4: "xlsx"}):
        return False
    repo_csv = ent_repocsv_path.get()
    issues_csv = ent_issuescsv_path.get()
    merged_path = ent_mergedxlsx_path.get()
    qf.merge_issues(repo_csv, issues_csv, merged_path)

# step 3
def generate_metrics():
    if not check_paths(not_empty=[5, 6, 7, 8], exists=[6, 7, 8], extensions={6: "xlsx", 7: "xlsx"}):
        return False
    domain_name = ent_domainame_path.get()
    similarity_matrix = ent_similaritytable_path.get()
    repositories_table = ent_repotable_path.get()
    metrics_output = ent_metrics_path.get()
    qf.main(similarity_matrix, repositories_table, domain_name, metrics_output)


window = tk.Tk()
window.title("Repositories Similarity & Quality Analyzer")

window.columnconfigure([0, 1], weight=1, minsize=5)
window.rowconfigure(0, weight=1, minsize=150)
window.rowconfigure(1, weight=1, minsize=200)
window.rowconfigure(2, weight=1, minsize=280)
# step 1 - Generate Similiarty Matrix
frame_step1 = tk.LabelFrame(master=window, padx=10, pady=5, text="Step 1: Generate Similarity Matrix")
lbl_repos = tk.Label(frame_step1, text="Location of repositories:")
ent_repos_path = tk.Entry(frame_step1, width=70)
btn_repos_path = tk.Button(frame_step1, text="Browse...", command=lambda: choose_directory(ent_repos_path))
lbl_output = tk.Label(frame_step1, text="Output location of similarity matrix:")
ent_output_path = tk.Entry(frame_step1, width=70)
btn_output_path = tk.Button(frame_step1, text="Browse...", command=lambda: save_file(ent_output_path, "xlsx"))
btn_generate = tk.Button(frame_step1, text="Generate Similarity Matrix", command=generate_similarity_matrix)

frame_step1.grid(row=0, column=0, padx=20, pady=(10, 5))
frame_step1.columnconfigure([0, 1], weight=1, minsize=30)
frame_step1.rowconfigure([0, 1, 2, 3, 4], weight=1, minsize=20)
lbl_repos.grid(row=0, column=0, columnspan=2, sticky="w")
ent_repos_path.grid(row=1, column=0)
btn_repos_path.grid(row=1, column=1)
lbl_output.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
ent_output_path.grid(row=3, column=0)
btn_output_path.grid(row=3, column=1)
btn_generate.grid(row=4, column=0, columnspan=2, pady=(10, 3))
# END of step 1 - Generate Similiarty Matrix

# Step 2 - Merge issues
frame_step2 = tk.LabelFrame(master=window, padx=10, pady=5, text="Step 2: Merge Issues")
lbl_repocsv = tk.Label(frame_step2, text="Choose the github_table_repository.csv file:")
ent_repocsv_path = tk.Entry(frame_step2, width=70)
btn_repocsv_path = tk.Button(frame_step2, text="Browse...", command=lambda: open_file(ent_repocsv_path))
lbl_issuescsv = tk.Label(frame_step2, text="Choose the github_table_issues.csv file:")
ent_issuescsv_path = tk.Entry(frame_step2, width=70)
btn_issuescsv_path = tk.Button(frame_step2, text="Browse...", command=lambda: open_file(ent_issuescsv_path))
lbl_mergedxlsx = tk.Label(frame_step2, text="Output location of merged Excel:")
ent_mergedxlsx_path = tk.Entry(frame_step2, width=70)
btn_mergedxlsx_path = tk.Button(frame_step2, text="Browse...", command=lambda: save_file(ent_mergedxlsx_path, "xlsx"))
btn_merge = tk.Button(frame_step2, text="Merge Issues.csv into Repository.csv", command=generate_merged_xlsx)

frame_step2.grid(row=1, column=0, padx=20, pady=(5, 5))
frame_step2.columnconfigure([0, 1], weight=1, minsize=30)
frame_step2.rowconfigure([0, 1, 2, 3, 4, 5, 6], weight=1, minsize=20)
lbl_repocsv.grid(row=0, column=0, columnspan=2, sticky="w")
ent_repocsv_path.grid(row=1, column=0)
btn_repocsv_path.grid(row=1, column=1)
lbl_issuescsv.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
ent_issuescsv_path.grid(row=3, column=0)
btn_issuescsv_path.grid(row=3, column=1)
lbl_mergedxlsx.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
ent_mergedxlsx_path.grid(row=5, column=0)
btn_mergedxlsx_path.grid(row=5, column=1)
btn_merge.grid(row=6, column=0, columnspan=2, pady=(10, 3))
# END of  Step 2 - Merge issues


# Step 3 - Generate Metrics
frame_step3 = tk.LabelFrame(master=window, padx=10, pady=5, text="Step 3: Generate Metrics")
lbl_domainame = tk.Label(frame_step3, text="Type the name of the domain:")
ent_domainame_path = tk.Entry(frame_step3, width=70)
lbl_similaritytable = tk.Label(frame_step3, text="Choose similiarity matrix file (genereated from step 1):")
ent_similaritytable_path = tk.Entry(frame_step3, width=70)
btn_similaritytable_path = tk.Button(frame_step3, text="Browse...", command=lambda: open_file(ent_similaritytable_path))
lbl_repotable = tk.Label(frame_step3, text="Choose repositories table file (generated from step 2):")
ent_repotable_path = tk.Entry(frame_step3, width=70)
btn_repotable_path = tk.Button(frame_step3, text="Browse...", command=lambda: open_file(ent_repotable_path))
lbl_metrics = tk.Label(frame_step3, text="Output location of quality metrics:")
ent_metrics_path = tk.Entry(frame_step3, width=70)
btn_metrics_path = tk.Button(frame_step3, text="Browse...", command=lambda: choose_directory(ent_metrics_path))
btn_metrics = tk.Button(frame_step3, text="Generate Metrics", command=generate_metrics)

frame_step3.grid(row=2, column=0, padx=20, pady=(5, 10))
frame_step3.columnconfigure([0, 1], weight=1, minsize=30)
frame_step3.rowconfigure([0, 1, 2, 3, 4, 5, 6, 7, 8], weight=1, minsize=10)
lbl_domainame.grid(row=0, column=0, columnspan=2, sticky="w")
ent_domainame_path.grid(row=1, column=0)
lbl_similaritytable.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
ent_similaritytable_path.grid(row=3, column=0)
btn_similaritytable_path.grid(row=3, column=1)
lbl_repotable.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
ent_repotable_path.grid(row=5, column=0)
btn_repotable_path.grid(row=5, column=1)
lbl_metrics.grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
ent_metrics_path.grid(row=7, column=0)
btn_metrics_path.grid(row=7, column=1)
btn_metrics.grid(row=8, column=0, columnspan=2, pady=(10, 3))
# END of Step 3 - Generate Metrics
window.maxsize(580, 740)
window.minsize(400, 620)
window.mainloop()
