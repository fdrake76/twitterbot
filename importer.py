import json
import os

from config import Config


class Importer:
    def __init__(self):
        self.config = Config()

    def import_from_json(self):
        filename = os.path.join(self.config.corpus_dir, self.config.json_file)
        with open(filename) as f:
            data = json.load(f)

        tweets = []
        for tweet_object in data:
            tweets.append(tweet_object['text'])

        print("Imported {} tweets from json file".format(len(tweets)))
        return tweets
