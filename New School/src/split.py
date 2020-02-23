import split_folders

# Cette librairie permet de facilement découper notre dossier src en 3 dossiers train / test / val
split_folders.ratio('../dataset/data/', output="../dataset/", seed=1337, ratio=(.5, .25, .25))
