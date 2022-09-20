#!/bin/python

####
# Designed for use with https://github.com/AUTOMATIC1111/stable-diffusion-webui
# Run this file from the root of the repo
#
# Usage in prompt: Put embedding name in prompt (using moxxi.pt and borderlands.pt)
# e.g. moxxi on a beach, in the style of borderlands sharp, clear lines, detailed
# or
# e.g. <moxxi> on a beach, in the style of <borderlands> sharp, clear lines, detailed
# <> are not required but can be used (and may give better results)
###

import re
import sys
import requests
from requests import get
import os.path
import json
import time
from bs4 import BeautifulSoup

# settings
settings = dict(
    concepts_library_url="https://huggingface.co/sd-concepts-library",
    embeddings_dir="./embeddings/",
    embeddings_dir_styles="./embeddings/style/",
    embeddings_dir_objects="./embeddings/object/",
    embeddings_samples_dir="./embeddings_samples/",
    ## allow list = None will download all available concepts
    allow_list=None,
    ## allow list = [...] will only download the list concepts
    ## example:
    #allow_list=["moxxi"],
    ## deny list will skip the listed concepts, the following gave parsing errors when loading
    deny_list=[
        "line-style",
        "faraon-love-shady",
        "floral",
        "ina-art",
        "osrsmini2",
        "spirithorse",
    ],
    download_images=False,
    max_images=4,
)

if not os.path.exists(settings['embeddings_dir']):
    os.makedirs(settings['embeddings_dir'])
    os.makedirs(settings['embeddings_dir_objects'])
    os.makedirs(settings['embeddings_dir_styles'])

if settings['download_images']:
    if not os.path.exists(settings['embeddings_samples_dir']):
        os.makedirs(settings['embeddings_samples_dir'])


def get_url_for_learned_embeddings(id):
    return f"https://huggingface.co/sd-concepts-library/{id}/resolve/main/learned_embeds.bin"
    
def get_url_for_learned_embeddings_type(id):
    return f"https://huggingface.co/sd-concepts-library/{id}/resolve/main/type_of_concept.txt"

def get_url_for_image_sample(id, n):
    return f"https://huggingface.co/sd-concepts-library/{id}/resolve/main/concept_images/{n}.jpeg"

def get_concept_type(id):
    response = get(get_url_for_learned_embeddings_type(id))
    return response.text

def get_concepts_library():
    print(f"Loading latest embedding repository list")
    sys.stdout.flush()
    page = requests.get(settings['concepts_library_url'])
    soup = BeautifulSoup(page.content, "html.parser")
    soup_models = soup.find(id="models")
    soup_parent = soup_models.parent
    soup_data = soup_parent['data-props']
    soup_data_json = json.loads(soup_data)
    soup_repos=soup_data_json["repos"]
    repo_count=len(soup_repos)
    repo_suffix=("" if repo_count == 1 else "s")
    print(f"Found {repo_count} repo{repo_suffix}")
    print(f"")
    sys.stdout.flush()
    return soup_repos

def download(url, file_name):
    with open(file_name, "wb") as file:
        response = get(url)
        file.write(response.content)

def url_exists(url):
    response = get(url)
    if response.status_code == 200:
        return True
    else:
        return False

def file_exists(file_name):
    return os.path.isfile(file_name) 

repos = get_concepts_library()
skipped_repos=0
downloaded_repos=0
already_downloaded_repos=0
for repo in repos:
    repo_id=repo["id"].replace("sd-concepts-library/","")
    if settings['allow_list'] is not None and repo_id not in settings['allow_list']:
        skipped_repos=skipped_repos+1
        continue
    if settings['deny_list'] is not None and repo_id in settings['deny_list']:
        skipped_repos=skipped_repos+1
        continue
    print(f"Processing {repo_id}")
    sys.stdout.flush()
    url=get_url_for_learned_embeddings(repo_id)
    concept_type = get_concept_type(repo_id)
    filename=f"{settings['embeddings_dir']}{concept_type}/{repo_id}.bin"
    if not file_exists(filename):
        print(f"  > Downloading {url} to {filename}")
        sys.stdout.flush()
        download(url,filename)
        downloaded_repos=downloaded_repos+1
    else:
        print(f"  > Already downloaded")
        already_downloaded_repos=already_downloaded_repos+1
        sys.stdout.flush()
    if settings["download_images"] == True:
        img_id=0
        no_more_images=False
        while not no_more_images:
            time.sleep(0.1)
            url=get_url_for_image_sample(repo_id, img_id)
            filename=f"{settings['embeddings_samples_dir']}{repo_id}.{img_id}.jpeg"
            if file_exists(filename):
                print(f"  > Already downloaded {img_id}.jpeg")
                sys.stdout.flush()
            else:
                if url_exists(url):
                    if not file_exists(filename):
                        print(f"  > Downloading {img_id}.jpeg to {filename}")
                        sys.stdout.flush()
                        download(url,filename)
                else:
                    no_more_images=True
            img_id=img_id+1
            if img_id >= settings['max_images']:
                no_more_images=True

print("")

repo_suffix=("" if downloaded_repos == 1 else "s")
print(f"Downloaded {downloaded_repos} repo{repo_suffix}")

repo_suffix=("" if already_downloaded_repos == 1 else "s")
print(f"Already downloaded {already_downloaded_repos} repo{repo_suffix}")

repo_suffix=("" if skipped_repos == 1 else "s")
print(f"Skipped {skipped_repos} repo{repo_suffix}")

print("")
print("Done.")