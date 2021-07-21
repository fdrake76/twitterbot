import re

from config import Config


class Vocabulary:
    """
    Encapsulates the vocabulary that is being used.  Every word that is trained is catalogued, along with
    special tokens indicating padding (for tensor inputs), start of sentence and end of sentence.
    """
    def __init__(self, sentences=None, pairs=None):
        self.sentences = [] if sentences is None else sentences
        self.pairs = [] if pairs is None else pairs
        self.config = Config()
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.config.PAD_token: "PAD", self.config.SOS_token: "SOS", self.config.EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD
        if len(sentences) > 0:
            print("Counting words...")
            for sentence in sentences:
                self.add_sentence(sentence)
            print("Counted words:", self.num_words)

    def add_sentence(self, sentence):
        # print("processing sentence {}".format(sentence))
        for word in sentence.split(' '):
            # print("adding word {}".format(word))
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def remove_underused_words(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('Removing {:.2%} of words that werent used at least {} times ({} out of {})'.format(
            1 - (len(keep_words) / len(self.word2index)), min_count, len(self.word2index) - len(keep_words),
            len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.config.PAD_token: "PAD", self.config.SOS_token: "SOS", self.config.EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.add_word(word)


class Preparer:
    def __init__(self):
        self.config = Config()

    def _is_sentence_under_max_size(self, pair):
        """
        Returns True if both sentences in a pair are under the MAX_WORDS config threshold, False otherwise
        """
        return len(pair[0].split(' ')) < self.config.max_words and len(pair[1].split(' ')) < self.config.max_words

    def _remove_pairs_beyond_max_size(self, pairs):
        """
        Returns all pairs from the input whose sentences are both under the config MAX_WORDS value
        """
        return [pair for pair in pairs if self._is_sentence_under_max_size(pair)]

    def clean_tweets(self, tweets):
        """
        Takes a list of tweets and performs sentence cleanup to better prepare for vocabulary and tensor input
        """
        cleaned_tweets = []
        for tweet in tweets:
            t = self.clean_input_tweet(tweet)
            if len(t) > 0:  # skip tweets who have been cleaned to the point of being empty
                cleaned_tweets.append(t)
        print("{} tweets after cleaning.".format(len(cleaned_tweets)))
        return cleaned_tweets

    @staticmethod
    def clean_input_tweet(tweet):
        """
        Takes a tweet and massages the input so it will be conducive to inhale into the vocabulary, or be used
        as tensor input.  Things such as separating out punctuation, numbers, etc and removing unnecessary noise
        such as hyperlinks and twitter pic links.
        """
        if tweet.startswith("RT "):  # Remove re-tweets
            return ""
        tweet = tweet.lower()  # Make all words lowercase for archiving into our vocabulary
        tweet = re.sub(r"pic.twitter.com/\w+", "", tweet)  # Strip twitter pics
        tweet = re.sub(r"http\S+", "", tweet)  # Strip URLs
        tweet = re.sub(r"@ (\w+)", r"@\1", tweet)  # Properly combine mentions
        tweet = re.sub(r"# (\w+)", r"#\1", tweet)  # Properly combine hashtags
        tweet = re.sub(r"[^\w.!?#@]+", r" ", tweet)  # Remove everything that isn't a word or . ! ? # @
        tweet = re.sub(r"\s+[.0-9]+\s+", r" ", tweet)  # Strip out standalone numbers
        tweet = re.sub(r"([.!?])", r" \1", tweet)  # Separate out . ! ? into their own "words"
        tweet = re.sub(r"\s+", r" ", tweet)  # Remove extra whitespaces so words are uniformly separated
        tweet = tweet.strip()  # Remove leading and trailing whitespaces
        return tweet

    @staticmethod
    def _create_tweet_conversations(tweets):
        """
        Creates a "conversation", which for this purpose will chain each of the tweets together as if it is
        one large conversation.
        """
        print("Building conversation...")
        # Split every line into pairs and normalize
        pairs = []
        for idx in range(len(tweets) - 1):
            pairs.append([tweets[idx], tweets[idx + 1]])
        return pairs

    def generate_conversation_pairs(self, sentences):
        """
        Take tweet sentences and chain them together into a "conversation", which is a list of lists, with each
        inner list being two dimensions.  The first element is the last element of the previous link in the chain.
        Example:

        [
           [ "hello", "well hi how are you" ],
           [ "well hi how are you", "i am good thank you" ],
           [ "i am good thank you", "you are very welcome" ]
        ]

        Since we are not having an actual conversation, like a typical chat bot, we will string all of the sentences
        in the inputted list together in this fashion, as if it was one very long conversation.  The conversation
        list will be truncated based on the MAX_WORDS config value, so we can control the upper bound input size
        for the neural network model.
        """
        print("Start preparing training data ...")
        pairs = self._create_tweet_conversations(sentences)

        print("Created {!s} sentence pairs".format(len(pairs)))
        pairs = self._remove_pairs_beyond_max_size(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        return pairs

    def trim_rare_words_from_voc(self, voc):
        """
        Updates the vocabulary object by removing references to words that are not used by a minimum number of times
        (set by min_count in the config)
        """
        voc.remove_underused_words(self.config.min_count)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in voc.pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(voc.pairs), len(keep_pairs),
                                                                    len(keep_pairs) / len(voc.pairs)))
        voc.pairs = keep_pairs

    def prepare_sentence_for_input(self, voc, sentence):
        """
        Takes an inputted sentence and formats it for valid input into the neural network.  Sending a sentence
        into the network without doing this could yield an error, because the network will barf if it sees a word
        that is not in its vocabulary.
        """

        if sentence == '':
            return ''

        words = [word for word in self.clean_input_tweet(sentence).split(' ') if word in voc.word2index]
        prepared_sentence = ' '.join(words) if len(words) > 0 else ''
        return prepared_sentence

    def clean_output_tweet(self, tweet):
        """
        Takes a sentence generated by the neural network and attempts to clean it a bit so it looks like a written
        sentence and not a bag of words put together.
        """

        tweet = re.sub(r" ([.!?])", r"\1", tweet)
        tweet = re.sub(r"n t ", r"n't ", tweet)
        tweet = re.sub(r"i ll ", r"i'll ", tweet)
        tweet = re.sub(r"u ll ", r"u'll ", tweet)
        tweet = re.sub(r"i m ", r"i'm ", tweet)
        tweet = re.sub(r"y re ", r"y're ", tweet)
        tweet = re.sub(r"(\w) s ", r"\1's ", tweet)

        # Shift verbiage of 1st person to 3rd person
        person = self.config.third_person_name
        tweet = re.sub(r"(^|\s)i am($|\s)", r"\1{} is\2".format(person), tweet)
        tweet = re.sub(r"(^|\s)i($|\s)", r"\1{}\2".format(person), tweet)
        tweet = re.sub(r"(^|\s)my($|\s)", r"\1{}'s\2".format(person), tweet)
        tweet = re.sub(r"(^|\s)me($|\s)", r"\1{}'s\2".format(person), tweet)
        tweet = re.sub(r"(^|\s)i'll($|\s)", r"\1{} will\2".format(person), tweet)

        tweet = tweet.capitalize()
        return tweet
