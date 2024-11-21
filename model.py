import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
from typing import List, Tuple
import random
from main import ChessHelper
from tqdm import tqdm


class ChessDataset(Dataset):
    def __init__(self, num_positions: int = 1000):
        self.num_positions = num_positions
        self.chess_helper = ChessHelper()
        self.positions = []
        self.labels = []
        self.generate_positions()

    def generate_positions(self):
        for _ in tqdm(range(self.num_positions), desc="Generating positions"):
            self.chess_helper.board.reset()

            num_moves = random.randint(5, 20)
            for _ in range(num_moves):
                legal_moves = list(self.chess_helper.board.legal_moves)
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                self.chess_helper.board.push(move)

            board_state = self.chess_helper.tokenize()
            legal_moves = self.chess_helper.legalMoves()

            self.positions.append(board_state)

            if legal_moves:
                best_move = random.choice(legal_moves)
                self.labels.append(best_move)
            else:
                continue

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.positions[idx]), self.labels[idx]


class ChessPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(19, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 64 * 64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.dropout(policy)
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=1)

        return policy


def square_name_to_index(square_name: str) -> int:
    """Convert a square name (e.g. 'e4') to an index 0-63."""
    file = ord(square_name[0]) - ord("a")
    rank = int(square_name[1]) - 1
    return rank * 8 + file


def move_to_index(move: Tuple[int, str, str]) -> int:
    """Convert a move tuple (piece_type, from_square, to_square) to a single index."""
    from_idx = square_name_to_index(move[1])
    to_idx = square_name_to_index(move[2])
    return from_idx * 64 + to_idx


def process_move(piece_type, from_square, to_square):
    """Process move components into a move index."""
    piece_type = piece_type.item()
    from_square = str(from_square)
    to_square = str(to_square)

    from_idx = square_name_to_index(from_square)
    to_idx = square_name_to_index(to_square)
    return from_idx * 64 + to_idx


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (boards, moves) in enumerate(train_loader):
            boards = boards.to(device)

            piece_types = moves[0]
            from_squares = moves[1]
            to_squares = moves[2]

            move_indices = []
            for p, f, t in zip(piece_types, from_squares, to_squares):
                try:
                    move_idx = process_move(p, f, t)
                    move_indices.append(move_idx)
                except Exception as e:
                    print(f"Error processing move: ({p}, {f}, {t})")
                    raise e

            move_indices = torch.tensor(move_indices).to(device)

            outputs = model(boards)
            loss = criterion(outputs, move_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")


def main():
    dataset = ChessDataset(num_positions=100)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    first_board, first_move = dataset[0]
    print("First move from dataset:", first_move)
    print(f"Move components: {first_move[0]}, {first_move[1]}, {first_move[2]}")

    model = ChessPolicy()
    train_model(model, train_loader, num_epochs=10)
    torch.save(model.state_dict(), "chess_model.pth")


if __name__ == "__main__":
    main()
