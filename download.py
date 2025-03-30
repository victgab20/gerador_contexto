import gdown

file_id = "1ktRVYv4v0Hd1nbyPlIXosBawt2aDalrn"
url = f"https://drive.google.com/uc?id={file_id}"
output = "train_data.csv"
gdown.download(url, output, quiet=False)
