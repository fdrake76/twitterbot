from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from config import Config


class Grapher:
    def __init__(self, session_name):
        self.config = Config()

        directory = os.path.join(self.config.save_dir, session_name)
        print("directory is {}".format(directory))
        if not os.path.exists(directory):
            print("Making dirs")
            os.makedirs(directory)

        self.writer = SummaryWriter(directory)
        self.start_training_time = datetime.now()

        # Save hyperparameters to TensorBoard
        hp_table = [
            "| {} | {} |".format(key, value)
            for key, value in self.config.__dict__.items()
        ]
        self.writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )

    def begin_training_timing(self):
        self.start_training_time = datetime.now()

    def write_loss(self, iteration, loss):
        self.writer.add_scalar("Iteration/Loss Per", loss, iteration - 1)

    def write_iteration(self, iteration):
        delta = datetime.now() - self.start_training_time
        self.writer.add_scalar("Iteration/Time Per (milliseconds)", iteration - 1,
                               int(delta.total_seconds() * 1000))
