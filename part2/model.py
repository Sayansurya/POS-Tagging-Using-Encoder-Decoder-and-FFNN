import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, encoder_input_dim, hidden_dim, num_layers):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = encoder_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim,
                            self.num_layers, batch_first=True, bidirectional = True)
        self.lstm = self.lstm.to(self.device)

    def forward(self, X):
        encoder_output, (encoder_hidden_state,
                         encoder_cell_state) = self.lstm(X)
        return encoder_output, encoder_hidden_state, encoder_cell_state


class Decoder(nn.Module):
    def __init__(self, decoder_input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_dim = decoder_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim,
                            self.num_layers, batch_first=True, bidirectional = True)
        self.linear = nn.Linear(2 * self.hidden_dim, self.output_dim)
        self.lstm = self.lstm.to(self.device)
        self.linear = self.linear.to(self.device)

    def forward(self, prev_timestep_output, encoder_hidden_state, encoder_cell_state):
        decoder_output, (decoder_hidden_state, decoder_cell_state) = self.lstm(
            prev_timestep_output, (encoder_hidden_state, encoder_cell_state))
        decoder_output = self.linear(decoder_output)
        return decoder_output


class Seq2SeqPOSTagger(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.encoder = Encoder(self.encoder_input_dim,
                               self.hidden_dim, self.num_layers)
        self.decoder = Decoder(
            self.decoder_input_dim, self.hidden_dim, self.output_dim, self.num_layers)

    def forward(self, X):
        encoder_output, encoder_hidden_state, encoder_cell_state = self.encoder(
            X)
        encoder_output = encoder_output.to(self.device)
        decoder_input = torch.cat((encoder_output[:, 0, :].unsqueeze(1), torch.zeros(X.shape[0], 1, self.output_dim).to(self.device)), dim = 2).to(self.device)
        decoder_output = torch.zeros(
            X.shape[0], X.shape[1], self.output_dim).to(self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_cell_state = encoder_cell_state
        for word_index in range(X.shape[1]):
            decoder_output_t = self.decoder(
                decoder_input, decoder_hidden_state, decoder_cell_state)
            decoder_output[:, word_index, :] = decoder_output_t.squeeze(1)
            decoder_input = torch.cat((encoder_output[:, word_index, :].unsqueeze(1), decoder_output[:, word_index, :].unsqueeze(1)), dim = 2)
            decoder_hidden_state = decoder_hidden_state.detach()
            decoder_cell_state = decoder_cell_state.detach()
        return decoder_output
    
    def predict(self, X):
        encoder_output, encoder_hidden_state, encoder_cell_state = self.encoder(
            X)
        encoder_output = encoder_output.to(self.device)
        decoder_input = torch.cat((encoder_output[:, 0, :].unsqueeze(1), torch.zeros(X.shape[0], 1, self.output_dim).to(self.device)), dim = 2).to(self.device)
        decoder_output = torch.zeros(
            X.shape[0], X.shape[1], self.output_dim).to(self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_cell_state = encoder_cell_state
        for word_index in range(X.shape[1]):
            decoder_output_t = self.decoder(
                decoder_input, decoder_hidden_state, decoder_cell_state)
            decoder_output[:, word_index, :] = decoder_output_t.squeeze(1)
            decoder_input = torch.cat((encoder_output[:, word_index, :].unsqueeze(1), decoder_output[:, word_index, :].unsqueeze(1)), dim = 2)
            decoder_hidden_state = decoder_hidden_state.detach()
            decoder_cell_state = decoder_cell_state.detach()
        return decoder_output
