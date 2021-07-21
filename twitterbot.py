from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random

from words import Preparer, Vocabulary
from model import Model
from batch import Batcher
from train import Trainer
from importer import Importer

tweet_importer = Importer()
preparer = Preparer()
batcher = Batcher()

# Load/Assemble voc and pairs
tweets = preparer.clean_tweets(tweet_importer.import_from_json())
voc = Vocabulary(tweets, preparer.generate_conversation_pairs(tweets))
model = Model(voc)

# Update vocabulary to trim words that don't meet the minimum count
preparer.trim_rare_words_from_voc(voc)

# Example for validation TODO
small_batch_size = 5
batches = batcher.batch_to_train_data(voc, [random.choice(voc.pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches
print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

trainer = Trainer(model)


def evaluateInput(voc):
    while 1:
        try:
            # Get input sentence
            input_sentence = input('(enter for a previously used tweet, your own sentence, or q/quit to quit)> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break

            # Normalize sentence
            input_sentence = preparer.prepare_sentence_for_input(voc, input_sentence)
            while input_sentence == '':
                input_sentence = random.choice(voc.sentences)
                input_sentence = preparer.prepare_sentence_for_input(voc, input_sentence)

            # Evaluate sentence
            output_words = trainer.evaluate(voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            output_sentence = preparer.clean_output_tweet(' '.join(output_words))
            print('Bot:', output_sentence)

        except KeyError:
            print("Error: Encountered unknown word.")


trainer.last_iteration = model.load_model(voc)
print("Starting Training!")
trainer.train(voc)

# Initialize search module
model.set_evaluate_mode()

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(voc)
