import torch
import chess
from main import ChessHelper
from model import ChessPolicy, square_name_to_index


class ChessGame:
    def __init__(self, model_path="chess_model.pth"):
        self.board = chess.Board()
        self.chess_helper = ChessHelper()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = ChessPolicy()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(self.device)

    def get_model_move(self):
        board_state = self.chess_helper.tokenize()
        board_tensor = torch.FloatTensor(board_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            move_probabilities = self.model(board_tensor).squeeze()

        legal_moves = list(self.board.legal_moves)
        legal_move_indices = []
        legal_move_probs = []

        for move in legal_moves:
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            move_idx = square_name_to_index(from_square) * 64 + square_name_to_index(
                to_square
            )
            legal_move_indices.append(move_idx)
            legal_move_probs.append(move_probabilities[move_idx].item())

        if not legal_moves:
            return None

        best_move_idx = legal_move_indices[torch.tensor(legal_move_probs).argmax()]
        from_square_idx = best_move_idx // 64
        to_square_idx = best_move_idx % 64

        for move in legal_moves:
            if move.from_square == from_square_idx and move.to_square == to_square_idx:
                return move

        return legal_moves[0]

    def play_game(self):
        print("Welcome to Chess! You play as White.")
        print("Enter moves in UCI format (e.g., 'e2e4')")
        print("Enter 'quit' to end the game\n")

        while not self.board.is_game_over():
            print("\nCurrent board position:")
            print(self.board)
            print("\nYour move (White):")

            while True:
                try:
                    move_str = input("> ").strip().lower()
                    if move_str == "quit":
                        return
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        break
                    else:
                        print("Illegal move! Try again.")
                except ValueError:
                    print("Invalid move format! Use UCI format (e.g., 'e2e4')")

            self.board.push(move)
            self.chess_helper.board = self.board

            if self.board.is_game_over():
                break

            print("\nAI is thinking...")

            ai_move = self.get_model_move()
            if ai_move is None:
                print("AI can't find a legal move!")
                break

            print(f"AI plays: {ai_move.uci()}")
            self.board.push(ai_move)
            self.chess_helper.board = self.board

        print("\nGame Over!")
        print(self.board)
        if self.board.is_checkmate():
            print("Checkmate!")
        elif self.board.is_stalemate():
            print("Stalemate!")
        elif self.board.is_insufficient_material():
            print("Draw due to insufficient material!")
        else:
            print("Draw!")


def main():
    game = ChessGame()
    game.play_game()


if __name__ == "__main__":
    main()
