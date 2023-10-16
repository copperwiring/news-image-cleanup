import sys, os, urllib.request
import shutil, glob
import pandas as pd
from tqdm import tqdm
import nltk
nltk.download('punkt')  

yyyy_mm = "2022-01"
base_dir_path = "/home/srishti/dev/news-images"
image_original_dir_path = os.path.join(base_dir_path, yyyy_mm)

# Monthly Folder with all subfolders of topics for each month
monthly_topic_dir = f"{yyyy_mm}_topics_data"
monthly_topic_dir_path = os.path.join(image_original_dir_path, monthly_topic_dir)

# Read all csv files and create one single csv file with two columns: uuid and maintext

# Read f"matcheduuid_{topic}.csv" csv file in the monthly_topic_dir_path
# and create a list of tuples with uuid and maintext

# Create a list of tuples with uuid and maintext
uuid_maintext_title = []
for topic in tqdm(os.listdir(monthly_topic_dir_path)):
    topic_dir_path = os.path.join(monthly_topic_dir_path, topic)
    csv_file_path = os.path.join(topic_dir_path, f"matcheduuid_{topic}.csv")
    df = pd.read_csv(csv_file_path)
    uuid_maintext_title.extend(df[['uuid', 'title', 'maintext']].values.tolist())

# Create a dataframe with uuid, title and maintext
df = pd.DataFrame(uuid_maintext_title, columns=['uuid', 'title', 'maintext'])
df.to_csv(os.path.join(base_dir_path, f"all_uuid_maintext_title.csv"), index=False)
print(f"Saved all text with no chunks at {os.path.join(base_dir_path, 'all_uuid_maintext_title.csv')}")

# Create a function to remove sentences (list) from the maintext
sentences_to_remove = ["USA TODAY", "CLICK HERE TO GET THE FOX NEWS APP", "Advertisement",
                       "Register now for FREE unlimited access to Reuters.com Register",
                       "Our Standards: The Thomson Reuters Trust Principles",
                       "Contributing: The Associated Press",
                       "Getty Images",
                       "Associated Press",
                       "Topline",
                       "CLICK HERE FOR MORE SPORTS COVERAGE ON FOXNEWS.COM",
                       "CLICK HERE TO GET FOX BUSINESS ON THE GO",
                       "CLICK HERE TO READ MORE ON FOX BUSINESS",
                       "Contact us at letters@time.com.",
                       "The Guardian is editorially independent. And we want to keep our journalism open and accessible to all. But we increasingly need our readers to fund our work.",
                       "SEARCH \"WIDER IMAGE\" FOR ALL STORIES",
                       "Email address By clicking Sign up, you agree to receive marketing emails from Insider as well as other partner offers and accept our Terms of Service and Privacy Policy",
                       "Over 3 million people read Morning Brew", "you should too Loading Something is loading.",
                       "GET FOX BUSINESS ON THE GO BY CLICKING HERE",
                       "Good morning and welcome to Fox News First. Here's what you need to know as you start your day"]
                       
def remove_selected_sentences_from_maintext(maintext, sentences_to_remove):
    """"
    Remove each sentence from the list of sentences_to_remove from the maintext

    Args:
        maintext (str): Text to remove sentences from
        sentences_to_remove (list): List of sentences to remove from maintext  

        Return:
            maintext (str): Text with sentences removed  
    """
    # remove emoji and other non-ascii characters
    maintext = maintext.encode("ascii", "ignore").decode() # 
    # Find any senten ce fom the list of sentences_to_remove is in each line of the maintext
    for sentence in sentences_to_remove:
        if sentence in maintext:
            # Remove the sentence from the maintext
            maintext = maintext.replace(sentence, "")
    return maintext

    
# Apply the function to each row in the DataFrame
for index, row in df.iterrows():
    df.at[index, 'maintext'] = remove_selected_sentences_from_maintext(row['maintext'], sentences_to_remove)

# Save the dataframe as a csv file
all_uuid_maintext_title_clean_file_name = f"all_uuid_maintext_title_removed_ads.csv"
df.to_csv(os.path.join(base_dir_path, f"{all_uuid_maintext_title_clean_file_name}"), index=False)
print(f"Saved all text with no chunks but removed useless news-ads at {os.path.join(base_dir_path, f'{all_uuid_maintext_title_clean_file_name}')}")
print(f"Number of rows in the clean csv file: {len(df)}")

# In df,  from 'maintext' column split text into 5 equal parts and create 4 new columns: chunk2, chunk3, chunk4, chunk5, chunk6
# Each new column will have approx 1/5th of the text till full stop (.) is encountered

# Create a new empty df and then columns for chunks
df_new = df.copy()

df_new['chunk2'] = ''
df_new['chunk3'] = ''
df_new['chunk4'] = ''
df_new['chunk5'] = ''
df_new['chunk6'] = ''


# Function to split text into chunks
def split_text_into_chunks(index, text):
    """
    """
    # remove emoji and other non-ascii characters
    text = text.encode("ascii", "ignore").decode() # 
    #Unsure if it ignores  numbers like 1.2, 2.3 etc and not on full stop (.) in abbreviations like U.S.A and ; etc
    sentences = nltk.tokenize.sent_tokenize(text, language='english')
    num_sentences = len(sentences)
    
    if num_sentences >= 5:
        # split sentences equally into 5 chunks
        equal_chunks = num_sentences // 5
        df_new.at[index, 'chunk2'] = " ".join(sentences[:equal_chunks])
        df_new.at[index, 'chunk3'] = " ".join(sentences[equal_chunks:2*equal_chunks])
        df_new.at[index, 'chunk4'] = " ".join(sentences[2*equal_chunks:3*equal_chunks])
        df_new.at[index, 'chunk5'] = " ".join(sentences[3*equal_chunks:4*equal_chunks])
        df_new.at[index, 'chunk6'] = " ".join(sentences[4*equal_chunks:])

       
    else:
        # Handle the case where there are fewer than 5 sentences. Then chunk1 will have all the text and 
        # rest will be have "Please skip to next page. Select any random annotation here. There is no penalty"
        df_new.at[index, 'chunk2'] = text
        df_new.at[index, 'chunk3'] = "Please skip to next page. Select any random annotation here. There is no penalty"
        df_new.at[index, 'chunk4'] = "Please skip to next page. Select any random annotation here. There is no penalty"
        df_new.at[index, 'chunk5'] = "Please skip to next page. Select any random annotation here. There is no penalty" 
        df_new.at[index, 'chunk6'] = "Please skip to next page. Select any random annotation here. There is no penalty"

    return df_new


# Apply the function to each row in the DataFrame
for index, row in df.iterrows():
    df_new = split_text_into_chunks(index, row['maintext'])


# Add title after remving rows with non empty column 8
# title column becomes chunk1
df_new['chunk1'] = df['title'].copy()

# Drop the title and maintext columns
df_new.drop(columns=['title', 'maintext'], inplace=True)

# Find row with non empty column 8th (so iloc 7) in df_new, delete it
# start_column = 7
# row_with_non_empty_column = df_new.iloc[:, start_column].notnull()
# print(f"Rows with non empty column {start_column}: {row_with_non_empty_column}")
# # Delete the rows in row_with_non_empty_column
# df_new = df_new[~row_with_non_empty_column]

# Save the dataframe as a csv file
chunks_file_name = f"all_uuid_maintext_title_chunks_removed_ads.csv"
df_new.to_csv(os.path.join(base_dir_path, f"{chunks_file_name}"), index=False)
print(f"Saved data with chunks at {os.path.join(base_dir_path, f'{chunks_file_name}')}")
print(f"Number of rows in chunks csv file: {len(df_new)}")

# Shuffle the dataframe and add seed value for reproducibility
df_new = df_new.sample(frac=1, random_state=42).reset_index(drop=True)

# Give 1200 rows from the shuffled dataframe
df_1200 = df_new[:1200]
file_name_1200 = f"1200_uuid_maintext_title_chunks_no_ads.csv"
df_1200.to_csv(os.path.join(base_dir_path, f"{file_name_1200}"), index=False)


# Read UUID value from df_1200 and create a list of uuids
# Select these UUIDs in "all_image_url_clean.csv" and create a new csv file with these UUIDs
# This new csv file will have uuid and image_url
# Read the csv file with uuid and image_url
df_image_url = pd.read_csv(os.path.join(base_dir_path, "all_image_url_clean.csv"))

# Create a list of uuids from df_1200
uuids_1200 = df_1200['uuid'].tolist()

# Create a new dataframe with uuid and image_url
df_image_url_1200 = df_image_url[df_image_url['uuid'].isin(uuids_1200)]

# Save the dataframe as a csv file
uuid_image_url_file_name = f"1200_uuid_image_url.csv"
df_image_url_1200.to_csv(os.path.join(base_dir_path, f'{uuid_image_url_file_name}'), index=False)
print(f"Saved {os.path.join(base_dir_path, f'{uuid_image_url_file_name}')}")
print(f"Number of rows in the image csv file: {len(df_image_url_1200)}")

# Create a new csv file with uuid, chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, image_url
# Use files: 1200_uuid_maintext_title_chunks.csv and 1200_uuid_image_url.csv
df_chunks = pd.read_csv(os.path.join(base_dir_path, f"{file_name_1200}"))
df_image_url = pd.read_csv(os.path.join(base_dir_path, f"{uuid_image_url_file_name}"))

# Merge the two dataframes on uuid in column sequence as: uuid, chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, image_url
df_merged = pd.merge(df_chunks, df_image_url, on='uuid')[['uuid', 'chunk1', 'chunk2', 'chunk3', 'chunk4', 'chunk5', 'chunk6', 'image_url']]
print(f"Number of rows BEFORE CLEANING in the merged csv file: {len(df_merged)}")
print("*"*100)

# df_merged.to_csv(os.path.join(base_dir_path, f"{merged_file_name}"), index=False)
# print(f"Saved chunks with image url BEFORE CLEANING at {os.path.join(base_dir_path, f'{merged_file_name}')}")

# ------------------------------------------------------------------------------------ ##

# For all uuids in df_merged, find all uuids not present in the all_images_clean folder. 
# Images are named as uuid.jpg

clean_image_dir_path = os.path.join(base_dir_path, "all_images_clean")
total_images = os.listdir(clean_image_dir_path)

# Create a list of uuids from the total_images
uuids_images = []
for image in total_images:
    uuid = image.split(".")[0]
    uuids_images.append(uuid)

# Create a list of uuids from df_merged
uuids_merged = df_merged['uuid'].tolist()

# Find the uuids in uuids_merged that are not in uuids_images
uuids_not_in_images = []
for uuid in uuids_merged:
    if uuid not in uuids_images:
        uuids_not_in_images.append(uuid)

print(f"Number of uuids in df_merged: {len(uuids_merged)}")
print(f"Number of uuids in uuids_images: {len(uuids_images)}")
print(f"Number of uuids in uuids_not_in_images: {len(uuids_not_in_images)}")
print(f"uuids not in images: {uuids_not_in_images}")

# Remove the rows from df_merged that have uuids not in images
df_merged = df_merged[~df_merged['uuid'].isin(uuids_not_in_images)]

# Save the dataframe as a csv file
merged_file_name = f"{len(df_merged)}_uuid_chunks_image_url.csv"
df_merged.to_csv(os.path.join(base_dir_path, f"{merged_file_name}"), index=False)
print(f"Saved chunks with image url AFTER CLEANING at {os.path.join(base_dir_path, f'{merged_file_name}')}")
print(f"Number of rows in the merged csv file: {len(df_merged)}")

# Download and save urls in df_merged['image_url'] to a new folder: all_images_{len(df_merged)}
# Create a new folder and delete if it already exists
all_images_folder_name = f"{len(df_merged)}_images"
all_images_folder_path = os.path.join(base_dir_path, f"{all_images_folder_name}")

if os.path.exists(all_images_folder_path):
    shutil.rmtree(all_images_folder_path)

os.mkdir(all_images_folder_path)

# Copy the images with name df_merged['uuid'].jpg from all_images_clean to all_images_{len(df_merged)}
for index, row in tqdm(df_merged.iterrows()):
    uuid = row['uuid']
    image_url = row['image_url']
    image_name = f"{uuid}.jpg"
    image_path = os.path.join(clean_image_dir_path, f"{image_name}")
    if os.path.exists(image_path):
        shutil.copy(image_path, all_images_folder_path)
    else:
        print(f"Image {image_name} not found in {clean_image_dir_path}")

print(f"Saved all images at {all_images_folder_path}")
print(f"Number of images in {all_images_folder_path}: {len(os.listdir(all_images_folder_path))}")

# ------------------------------------------------------------------------------------ ##
# Split the merged csv file into 6 csv files with 200 rows each and save in anew folder: 1200_uuid_chunks_image_url_splits
# Use the merged csv file created in the previous step

# Create a new folder and delete if it already exists
splits_folder_name = f"{len(df_merged)}_uuid_chunks_image_url_splits"
splits_folder_path = os.path.join(base_dir_path, f"{splits_folder_name}")
if os.path.exists(splits_folder_path):
    shutil.rmtree(splits_folder_path)

os.mkdir(splits_folder_path)

# Remeber data is already shuffled

split_folder_name = f"{len(df_merged)}_uuid_chunks_image_url_splits" 
split_folder_path = os.path.join(base_dir_path, f"{split_folder_name}")

# Delete folder if it already exists
if os.path.exists(split_folder_path):
    shutil.rmtree(split_folder_path)

os.mkdir(split_folder_path)

# Split the dataframe into 6 csv files with 200 rows each
for i in tqdm(range(6)):
    start = i * 200
    end = (i+1) * 200
    df_merged[start:end].to_csv(os.path.join(split_folder_path, f"split_{i}_uuid_chunks_image_url.csv"), index=False) # i+1 because 1 is title
    print(f"Saved chunks+images at {split_folder_path}/split_{i}_uuid_chunks_image_url.csv")

print(f"Saved all chunks+images at {split_folder_path}")
print("*"*100)