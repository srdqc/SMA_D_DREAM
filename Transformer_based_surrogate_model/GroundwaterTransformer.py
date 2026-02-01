import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from model.load_data import get_data
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.optim.lr_scheduler import StepLR


def mean_squared_error(y_true, y_pred):
    """Compute Mean Squared Error (MSE)."""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_relative_error(y_true, y_pred):
    """Compute Mean Relative Error (MRE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true))


def load_and_preprocess_data(directory_path, data_line, test_size=0.4):
    """
    Load data from directory, split into training and validation sets,
    and apply standardization.

    Parameters:
    directory_path (str): Path to data directory
    data_line (int): Number of samples
    test_size (float): Proportion of validation data

    Returns:
    Tensors and fitted scalers
    """
    param_train, param_val, result_train, result_val = get_data(
        directory_path, data_line, test_size
    )

    # Standardize input parameters
    scaler_param = StandardScaler()
    param_train_scaled = scaler_param.fit_transform(param_train)
    param_val_scaled = scaler_param.transform(param_val)

    # Standardize output results
    scaler_result = StandardScaler()
    result_train_scaled = scaler_result.fit_transform(result_train)
    result_val_scaled = scaler_result.transform(result_val)

    # Convert to PyTorch tensors
    train_inputs = torch.FloatTensor(param_train_scaled)
    train_targets = torch.FloatTensor(result_train_scaled)
    val_inputs = torch.FloatTensor(param_val_scaled)
    val_targets = torch.FloatTensor(result_val_scaled)

    return (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        scaler_param,
        scaler_result,
    )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a trainable parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Parameters:
        x (Tensor): Input tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[: x.size(1), :]
        return x


class GroundwaterTransformer(nn.Module):
    """
    Transformer-based surrogate model for groundwater simulation.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=1024,
        nhead=64,
        dim_feedforward=512,
        num_layers=6,
    ):
        super().__init__()

        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layers, num_layers)

        # Transformer decoder
        decoder_layers = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.decoder = TransformerDecoder(decoder_layers, num_layers)

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        # Output projection
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        """
        Forward pass.

        Parameters:
        src (Tensor): Input tensor [batch_size, 1, input_dim]

        Returns:
        Tensor: Output predictions [batch_size, output_dim]
        """
        src = self.embedding(src)
        src = self.pos_encoder(src)

        # Encode input sequence
        memory = self.encoder(src)

        # Prepare query for decoder
        query = self.query.repeat(src.size(0), 1, 1)
        query = self.pos_encoder(query)

        # Decode and project to output space
        output = self.decoder(query, memory)
        output = self.output_layer(output.squeeze(1))

        return output


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scaler_result,
    epochs=10,
):
    """
    Train the model and evaluate on validation set.

    Returns:
    Best training loss, validation loss, training R², validation R²
    """
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = None
    best_train_r2 = None
    best_val_r2 = None

    for epoch in range(epochs):
        model.train()
        train_predictions = []
        train_actuals = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_predictions.extend(outputs.detach().cpu().numpy())
            train_actuals.extend(targets.cpu().numpy())

        # Inverse standardization
        train_predictions = scaler_result.inverse_transform(train_predictions)
        train_actuals = scaler_result.inverse_transform(train_actuals)

        train_loss = loss.item()
        train_losses.append(train_loss)
        train_r2 = r2_score(train_actuals, train_predictions)
        train_r2_scores.append(train_r2)

        model.eval()
        val_predictions = []
        val_actuals = []
        total_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()

                val_predictions.extend(outputs.cpu().numpy())
                val_actuals.extend(targets.cpu().numpy())

        # Inverse standardization
        val_predictions = scaler_result.inverse_transform(val_predictions)
        val_actuals = scaler_result.inverse_transform(val_actuals)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_r2 = r2_score(val_actuals, val_predictions)
        val_r2_scores.append(val_r2)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Train R²: {train_r2:.4f} | "
            f"Val R²: {val_r2:.4f}"
        )

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_train_r2 = train_r2
            best_val_r2 = val_r2

    print(
        f"Best model at epoch {best_epoch} | "
        f"Train Loss: {best_train_loss:.4f} | "
        f"Val Loss: {best_val_loss:.4f} | "
        f"Train R²: {best_train_r2:.4f} | "
        f"Val R²: {best_val_r2:.4f}"
    )

    return best_train_loss, best_val_loss, best_train_r2, best_val_r2


def rmse_loss(y_pred, y_true):
    """RMSE loss function."""
    return torch.sqrt(nn.MSELoss()(y_pred, y_true))


def main(data_line, j):
    """
    Run a single training experiment.
    """
    directory_path = "Nash_Coefficient"
    test_size = 0.4
    data_number = round(data_line / 0.6 / 8)
    batch_size = 64

    (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        scaler_param,
        scaler_result,
    ) = load_and_preprocess_data(directory_path, data_number, test_size)

    input_dim = train_inputs.shape[1]
    output_dim = train_targets.shape[1]

    train_dataset = TensorDataset(train_inputs.unsqueeze(1), train_targets)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = TensorDataset(val_inputs.unsqueeze(1), val_targets)
    val_loader = DataLoader(val_dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    d_model = 512
    nhead = 8
    dim_feedforward = 1024
    epochs = 100
    num_layers = j + 1

    model = GroundwaterTransformer(
        input_dim,
        output_dim,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
    ).to(device)

    criterion = rmse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Record training time
    start_time = time.time()
    best_train_loss, best_val_loss, best_train_r2, best_val_r2 = train_and_evaluate(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scaler_result,
        epochs,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    return (
        best_train_loss,
        best_val_loss,
        best_train_r2,
        best_val_r2,
        elapsed_time,
    )


if __name__ == "__main__":
    for i in range(0, 10):
        for j in range(0, 6):

            # Example data size list
            data_lines = [10000]

            for data_line in data_lines:
                print(f"Testing with data_line: {data_line}")

                (
                    best_train_loss,
                    best_val_loss,
                    best_train_r2,
                    best_val_r2,
                    elapsed_time,
                ) = main(data_line, j)

                print(
                    f"Execution time for data_line {data_line}: "
                    f"{elapsed_time:.2f} seconds"
                )

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(
                    "training_results_transformerlayer.txt",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        f"Run {i + 1}: Data {data_line}, "
                        f"{j + 1} Layers, {current_time}; "
                        f"Execution time: {elapsed_time:.2f} s; "
                        f"Epochs 100/100 | "
                        f"Train Loss: {best_train_loss:.4f} | "
                        f"Val Loss: {best_val_loss:.4f} | "
                        f"Train R²: {best_train_r2:.4f} | "
                        f"Val R²: {best_val_r2:.4f}\n"
                    )
