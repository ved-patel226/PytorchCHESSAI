from model import ChessDataset


def test_chess_dataset():
    chess_dataset = ChessDataset(num_positions=100)

    first_board, first_move = chess_dataset[0]
    print("First move from dataset:", first_move)


def main() -> None:
    test_chess_dataset()


if __name__ == "__main__":
    main()
