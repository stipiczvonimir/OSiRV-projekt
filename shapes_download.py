import kagglehub
import shutil

default_path = kagglehub.dataset_download("smeschke/four-shapes")

custom_path = "./shapes"

shutil.move(default_path, custom_path)
print("Dataset moved to:", custom_path)
