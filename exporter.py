import json
import time
from datetime import datetime, timedelta
import GetOldTweets3 as got
import argparse

_datefmt = '%Y-%m-%d'


def _export_tweet_chunk(username, start_date=None, end_date=None):
    tweet_criteria = got.manager.TweetCriteria().setUsername(username)
    if start_date is not None:
        tweet_criteria = tweet_criteria.setSince(start_date)
    if end_date is not None:
        tweet_criteria = tweet_criteria.setUntil(end_date)
    tweets = []
    for tweet in got.manager.TweetManager.getTweets(tweet_criteria):
        tweets.append({'date': tweet.date.isoformat(), 'text': tweet.text})

    return tweets


def _add_to_date(start_date, days_to_add):
    date_obj = datetime.strptime(start_date, _datefmt) + timedelta(days=days_to_add)
    return date_obj.strftime(_datefmt)


class Exporter:
    def __init__(self, days_per_chunk=90, indent=0, pause=None):
        self.days_per_chunk = days_per_chunk
        self.indent = indent
        self.pause = pause

    def export_from_twitter(self, username, start_date='2006-03-21', end_date=datetime.now().strftime(_datefmt)):
        iter_start_date = start_date
        iter_end_date = _add_to_date(iter_start_date, self.days_per_chunk)
        tweet_data = []
        print("Pulling tweet info for user {} from {} up to {}...".format(username, start_date, end_date))
        while datetime.strptime(iter_end_date, _datefmt) <= datetime.strptime(end_date, _datefmt):
            print("Pulling a chunk of tweets from {} up to {}".format(iter_start_date, iter_end_date))
            tweet_data.extend(_export_tweet_chunk(username, iter_start_date, iter_end_date))
            iter_start_date = iter_end_date
            iter_end_date = _add_to_date(iter_start_date, self.days_per_chunk)
            if self.pause is not None:
                print("...sleeping {} seconds...".format(self.pause))
                time.sleep(self.pause)

        return tweet_data

    @staticmethod
    def export_from_json_file(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        print("Pulled {} tweets from file {}".format(len(data), json_file))
        return data

    def write_to_json_file(self, data, output_file):
        print("Writing {} tweets to file {}".format(len(data), output_file))
        with open(output_file, 'w') as file:
            json.dump(data, file, indent=self.indent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, help='the twitter username')
    parser.add_argument('--start', type=str, help='the start date (inclusive) in yyyy-mm-dd format.  '
                                                  'Defaults to 2006-03-21 (the date of the first ever tweet)',
                        default='2006-03-21')
    parser.add_argument('--end', type=str, help='the end date (exclusive) in yyyy-mm-dd format.  '
                                                'Defaults to the current date.',
                        default=datetime.now().strftime(_datefmt))
    parser.add_argument('--outfile', type=str, help='the output file')
    parser.add_argument('--infile', type=str, help='the input file to load before appending.  '
                                                   'By default, there is no file to load.',
                        default=None)
    parser.add_argument('--step', type=int, help='the number of days advanced in each increment.  '
                                                 'Defaults to 30 days.',
                        default=30)
    parser.add_argument('--indent', type=int, help='the number of indent spaces when outputting JSON.  '
                                                   'Defaults to 0 (no indent formatting)',
                        default=0)
    parser.add_argument('--pause', type=int, help='the number of seconds to pause in between each chunk, '
                                                  'to prevent hitting the Twitter rate limit.  Defaults to no pause.',
                        default=None)
    args = parser.parse_args()

    exporter = Exporter(args.step, args.indent, args.pause)
    original_data = exporter.export_from_json_file(args.infile) if args.infile is not None else []
    new_tweet_data = exporter.export_from_twitter(args.username, args.start, args.end)
    original_data.extend(new_tweet_data)
    print("Sorting {} tweets by date...".format(len(original_data)))
    original_data.sort(key=lambda x: x['date'], reverse=True)
    exporter.write_to_json_file(original_data, args.outfile)
