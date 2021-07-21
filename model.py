import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os


from config import Config


class EncoderRNN(nn.Module):
    def __init__(self, embedding):
        super(EncoderRNN, self).__init__()
        self.config = Config()
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, self.config.encoder_n_layers,
                          dropout=(0 if self.config.encoder_n_layers == 1 else self.config.dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.config.hidden_size] + outputs[:, :, self.config.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, embedding, output_size):
        super(LuongAttnDecoderRNN, self).__init__()
        self.config = Config()

        # Keep for reference
        self.output_size = output_size

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, self.config.decoder_n_layers,
                          dropout=(0 if self.config.decoder_n_layers == 1 else self.config.dropout))
        self.concat = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, output_size)

        self.attn = Attn(self.config.attn_model, self.config.hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = Config()

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.config.decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.config.device(), dtype=torch.long) * self.config.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.config.device(), dtype=torch.long)
        all_scores = torch.zeros([0], device=self.config.device())
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


class Model:
    def __init__(self, voc):
        self.config = Config()

        print('Building models from established vocabulary...')
        embedding = nn.Embedding(voc.num_words, self.config.hidden_size)
        self.encoder = EncoderRNN(embedding).to(self.config.device())
        self.decoder = LuongAttnDecoderRNN(embedding, voc.num_words).to(self.config.device())
        self.searcher = GreedySearchDecoder(self.encoder, self.decoder)
        self.encoder.train()
        self.decoder.train()
        print('Models built.  Building optimizers ...')
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.config.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=self.config.learning_rate * self.config.decoder_learning_ratio)
        print('Optimizers built.')

    def load_model(self, voc):
        if self.config.load_filename is None:
            return 0

        filename = os.path.join(self.config.save_dir, self.config.load_filename)
        print('Loading model from {}'.format(filename))
        checkpoint = torch.load(filename, map_location=self.config.device())
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        voc.__dict__ = checkpoint['voc_dict']

        return checkpoint['iteration']

    def save_model(self, voc, session_name, iteration, loss):
        directory = os.path.join(self.config.save_dir, session_name)
        filename = os.path.join(directory, '{}_{}_{}.tar'.format(self.config.model_name, 'checkpoint', iteration))
        print("Saving checkpoint {} at {}".format(iteration, filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'session_name': session_name,
            'iteration': iteration,
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
        }, filename)

    def set_evaluate_mode(self):
        self.encoder.eval()
        self.decoder.eval()
