import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

repo_name = {}


def read_data(filepath):
    d = pd.read_csv(filepath)
    return d


def clean_data():
	curr_millis = int(round(time.time() * 1000))

	d = read_data("watcher.csv")
	repo = d["repo_name"]
	watch_count = d["watch_count"]
	for i in range(len(repo)):
		repo_name[repo[i]] = [watch_count[i], np.nan, np.nan, np.nan, np.nan, np.nan]


	d = read_data("lang_count.csv")
	repo = d["L_repo_name"]
	lang_count = d["L_lang_count"]
	mean_lang_count = np.mean(np.array(lang_count))
	for i in range(len(repo)):
		if repo[i] in repo_name:
			repo_name[repo[i]][1] = lang_count[i]


	d = read_data("files_count.csv")
	repo = d["S_repo_name"]
	file_count = d["C_files"]
	mean_file_count = np.mean(np.array(file_count))
	for i in range(len(repo)):
		if not pd.isnull(file_count[i]):
			repo_name[repo[i]][2] = file_count[i]

	d = read_data("committer_count.csv")
	repo = d["S_repo_name"]
	committer_count = d["C_committer_count"]
	mean_committer_count = np.mean(np.array(committer_count))
	for i in range(len(repo)):
		if not pd.isnull(committer_count[i]):
			repo_name[repo[i]][3] = committer_count[i]

	d = read_data("commit_count.csv")
	repo = d["S_repo_name"]
	commit_count = d["C_comm_count"]
	mean_commit_count = np.mean(np.array(commit_count))
	for i in range(len(repo)):
		if not pd.isnull(commit_count[i]):
			repo_name[repo[i]][4] = commit_count[i]

	d = read_data("committer_date.csv")
	repo = d["S_repo_name"]
	commit_date = d["C_commit_date"]
	for i in range(len(repo)):
		if not pd.isnull(commit_date[i]):
			dt_obj = datetime.strptime(commit_date[i][:-4],'%Y-%m-%d %H:%M:%S')
			millisec = dt_obj.timestamp() * 1000
			repo_name[repo[i]][5] =  curr_millis - millisec

	repo_name1 = repo_name.copy()
	print(len(repo_name1))
	for i in repo_name.keys():
		if pd.isnull(repo_name[i][5]):
			del repo_name1[i]

	print(len(repo_name1))

	df = pd.DataFrame.from_dict(repo_name1, orient='index')
	values = {'1': mean_lang_count, '2': mean_file_count, '3': mean_committer_count, '4': mean_commit_count}
	df.fillna(value=values)
	Frame=pd.DataFrame(df.values, columns = ["repo_name", "watch_count", "lang_count", "files_count", "committer_count", "commit_count", "commit_date"])
	Frame.to_csv("Final.csv")

clean_data()

