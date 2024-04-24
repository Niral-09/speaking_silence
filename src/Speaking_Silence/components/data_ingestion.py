import os
import json
import requests
from mongoengine import connect
from Speaking_Silence import logger
from Speaking_Silence.models import SignInstance, SignInstanceEmbedded
from Speaking_Silence.utils.common import get_size
from Speaking_Silence.entity.config_entity import DataIngestionConfig
from requests.exceptions import ConnectionError, HTTPError, Timeout, RequestException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_and_store_data(self):
        with open(self.config.json_file, 'r') as f:
            data = json.load(f)
        
        # Connect to MongoDB
        logger.info(f"Connecting to MongoDB at {self.config.db_host}...")
        connect(self.config.db_name, host=self.config.db_host)

        # Iterate over each entry in the JSON file
        logger.info(f"Downloading data")
        for entry in data:
            gloss = entry['gloss']
            if gloss not in ["all", "computer", "before", "cool"]:
                continue 
            logger.info(f"Downloading data for gloss {gloss}...")
            instances = entry['instances']
            
            gloss_folder = os.path.join(self.config.artifact_folder, gloss)
            if not os.path.exists(gloss_folder):
                os.makedirs(gloss_folder)
            
            # Store metadata in MongoDB
            sign_instance = SignInstance(gloss=gloss, instances=[])
            headers = {
                'User-Agent': self.config.user_agent
            }
            # Iterate over each instance of the sign
            for instance in instances:
                try:
                    if instance['source'] == "aslpro":
                        continue
                    # Download the video
                    video_url = instance['url']
                    video_name = f"{gloss}_{instance['video_id']}.mp4"
                    video_path = os.path.join(gloss_folder, video_name)
                    if not os.path.exists(video_path):
                        response = requests.get(video_url, headers=headers)
                        with open(video_path, 'wb') as video_file:
                            video_file.write(response.content)
                        
                        # Add metadata to the MongoDB document
                        sign_instance.instances.append(SignInstanceEmbedded(
                            gloss = gloss,
                            filename = video_name,
                            bbox = instance['bbox'],
                            fps = instance['fps'],
                            video_url = video_url,
                            video_id = instance['video_id']
                        ))
                except (ConnectionError, HTTPError, Timeout, RequestException) as e:
                    print(f"Error downloading video for gloss '{gloss}': {str(e)}")
                    continue
            
            # Save the sign instance document to MongoDB
            if not sign_instance.instances == []:
                sign_instance.save()
            logger.info(f"Sign instance document for gloss '{gloss}' saved to MongoDB")
