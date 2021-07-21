import torch
import os


class Config:
    def __init__(self):
        self.load_filename = None  # The filename of the checkpoint to load from relative to the save directory, or None if starting from scratch
        self.json_file = "realDonaldTrump.json"  # The JSON file in the data corpus directory containing tweets to import
        self.third_person_name = 'Trump'  # Name that is used when converting 1st person vocabulary to 3rd person
        self.max_words = 60  # Maximum sentence length to consider
        self.min_count = 3   # Words used less than this number of times are stripped from the vocabulary

        # Configure training/optimization
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.n_iteration = 4000  # Total number of iterations to run
        self.save_every = 500  # Saves a checkpoint at this iteration multiple

        # Configure models
        self.model_name = 'tb_model'  # The name of the model, used when saving the checkpoint files
        self.attn_model = 'dot'
        # attn_model = 'general'
        # attn_model = 'concat'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64  # Size of the sampling batches

        # You shouldn't need to change these values
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # data dir
        self.save_dir = os.path.join(self.data_dir, "save")  # save dir
        self.corpus_dir = os.path.join(self.data_dir, "corpus")  # corpus dir
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"  # Setting capability for CPU/GPU
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token
        self.print_every = 1  # Prints iteration statistics at this iteration multiple

    def device(self):
        return torch.device(self.device_name)
