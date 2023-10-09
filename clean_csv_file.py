import sys, os, urllib.request
import shutil, glob
import pandas as pd
from tqdm import tqdm

## ------------------------------------------------------------------------------------ ##
# PART 0: Copy all the images from the subfolders of the monthly_topic_dir_path to a new folder
# 
# Read files from the new folder and create a csv with two columns: 
# # a) uuid: name of the image file without extension
# # b) image_path: "https://github.com/copperwiring/news-images/blob/main/image_dir/<image_name>"

yyyy_mm = "2022-01"
base_dir_path = "/home/srishti/dev/news-images"
image_original_dir_path = os.path.join(base_dir_path, yyyy_mm)

# Monthly Folder with all subfolders of topics for each month
monthly_topic_dir = f"{yyyy_mm}_topics_data"
monthly_topic_dir_path = os.path.join(image_original_dir_path, monthly_topic_dir)

allimage_dir_path = os.path.join(base_dir_path, "all_images")

# ------------------------------------------------------------------------------------ ##
# # Delete the allimage_dir_path if it already exists and create a new one
# if os.path.exists(allimage_dir_path):
#     print(f"Deleted {allimage_dir_path} because it already exists")
#     shutil.rmtree(allimage_dir_path)
# os.mkdir(allimage_dir_path)
# print(f"Created {allimage_dir_path}")


# # Read all the images in the subfolders of monthly_topic_dir_path
# # and copy it to the allimage_dir_path
# print(f"Reading images from {monthly_topic_dir_path} and copying it to {allimage_dir_path}")
# for topic in tqdm(os.listdir(monthly_topic_dir_path)):
#     topic_dir_path = os.path.join(monthly_topic_dir_path, topic)
#     image_dir_path = os.path.join(topic_dir_path, "images")
#     for image in os.listdir(image_dir_path):
#         image_path = os.path.join(image_dir_path, image)
#         shutil.copy(image_path, allimage_dir_path)

# ------------------------------------------------------------------------------------ ##

# Read the number of images in the new folder
total_images = os.listdir(allimage_dir_path)
print(f"Total number of images in {allimage_dir_path}: {len(total_images)}")

# Read the number of images in the original folder
original_total_images = glob.glob(os.path.join(image_original_dir_path, "**/*.jpg"), recursive=True)
print(f"Total number of images in {image_original_dir_path}: {len(original_total_images)}")


## ---------------------------------------------------------------------------------------------------- ##
# PART 1: Create a csv file with two columns: uuid and image_path for all the images in the new folder
## ---------------------------------------------------------------------------------------------------- ##

# Create a list of tuples with uuid and image_path
uuid_image_path = []
prefix_file_path = "https://raw.githubusercontent.com//copperwiring/news-images/main/image_dir/"
for image in tqdm(total_images):
    uuid = image.split(".")[0]
    image_path = os.path.join(prefix_file_path, image)
    uuid_image_path.append((uuid, image_path))

# Create a df from the list of tuples
df = pd.DataFrame(uuid_image_path, columns=["uuid", "image_url"])

# Save the df as csv
csv_file_path = os.path.join(base_dir_path, "all_image_url.csv")
print(f"Saving csv file to {csv_file_path}")

# Delete the csv file if it already exists
if os.path.exists(csv_file_path):
    print(f"Deleted {csv_file_path} because it already exists")
    os.remove(csv_file_path)
df.to_csv(csv_file_path, index=False)
print(f"Saved csv file to {csv_file_path}")

# ------------------------------------------------------------------------------------ ##
PART 2: Check if the uuid in the cleaned_csv file in each topic folder is present in the uuids in the all_image_url.csv file
uuids in all_image_url.csv file belong to images which have been checked to be of good quality
# ------------------------------------------------------------------------------------ ##

list_of_uuid = df["uuid"].tolist()

for topic in tqdm(os.listdir(monthly_topic_dir_path)):
    # Open the csv file that starts with "cleaned_" and check if the "uuid" column values
    # are present in the list_of_uuid
    # if not present, delete the row from the csv file in the monthly_topic_dir_path
    topic_dir_path = os.path.join(monthly_topic_dir_path, topic)
    csv_file_path = os.path.join(topic_dir_path, f"cleaned_{topic}.csv")
    df = pd.read_csv(csv_file_path)
    df = df[df["uuid"].isin(list_of_uuid)]
    # new csv file = "matcheduuid_" + topic
    new_csv_file_path = os.path.join(topic_dir_path, f"matcheduuid_{topic}.csv")
    df.to_csv(new_csv_file_path, index=False)

# Count the number of rows in all the new csv files in the monthly_topic_dir_path
# and print the total number of rows
total_rows = 0
for topic in os.listdir(monthly_topic_dir_path):
    topic_dir_path = os.path.join(monthly_topic_dir_path, topic)
    csv_file_path = os.path.join(topic_dir_path, f"matcheduuid_{topic}.csv")
    df = pd.read_csv(csv_file_path)
    total_rows += len(df)

print(f"Total number of rows in all the new csv files in {monthly_topic_dir_path}: {total_rows}")


