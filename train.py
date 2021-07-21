import torch
import torch.nn as nn
import random
from datetime import datetime

from config import Config
from batch import Batcher
from graph import Grapher


class Trainer:
    def __init__(self, model):
        self.model = model
        self.config = Config()
        self.batcher = Batcher()
        self.last_iteration = 0

    def mask_nll_loss(self, inp, target, mask):
        n_total = mask.sum()
        cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(self.config.device())
        return loss, n_total.item()

    def _train_iteration(self, input_variable, lengths, target_variable, mask, max_target_len):
        # Zero gradients
        self.model.encoder_optimizer.zero_grad()
        self.model.decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(self.config.device())
        lengths = lengths.to(self.config.device())
        target_variable = target_variable.to(self.config.device())
        mask = mask.to(self.config.device())

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.model.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.config.SOS_token for _ in range(self.config.batch_size)]])
        decoder_input = decoder_input.to(self.config.device())

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.config.decoder_n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.config.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.model.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, n_total = self.mask_nll_loss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.model.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.config.batch_size)]])
                decoder_input = decoder_input.to(self.config.device())
                # Calculate and accumulate loss
                mask_loss, n_total = self.mask_nll_loss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.config.clip)
        _ = torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), self.config.clip)

        # Adjust model weights
        self.model.encoder_optimizer.step()
        self.model.decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def train(self, voc):
        session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print('Initializing session {}...'.format(session_name))
        grapher = Grapher(session_name)

        # Load batches for each iteration
        training_batches = [
            self.batcher.batch_to_train_data(voc, [random.choice(voc.pairs) for _ in range(self.config.batch_size)])
            for _ in range(self.config.n_iteration)
        ]

        # Initializations
        print_loss = 0
        start_iteration = self.last_iteration + 1

        # Training loop
        print("Run tensorboard --logdir ./data/save and go to http://localhost:6006/ to see in real time "
              "the training performance.")
        print("Training for {} iterations...".format(self.config.n_iteration))
        grapher.begin_training_timing()
        for iteration in range(start_iteration, self.config.n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = self._train_iteration(input_variable, lengths, target_variable, mask, max_target_len)
            print_loss += loss

            # Print progress
            if iteration % self.config.print_every == 0:
                print_loss_avg = print_loss / self.config.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / self.config.n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if iteration % self.config.save_every == 0:
                self.model.save_model(voc, session_name, iteration, loss)
            grapher.write_iteration(iteration)
            grapher.write_loss(iteration, loss)

    def evaluate(self, voc, sentence):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [self.batcher.indexes_from_sentence(voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.config.device())
        lengths = lengths.to(self.config.device())
        # Decode sentence with searcher
        tokens, scores = self.model.searcher(input_batch, lengths, self.config.max_words)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words
