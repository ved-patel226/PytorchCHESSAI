from tqdm import tqdm
import numpy as np
import chess.pgn


def load_pgn(file_path):
    ttl_moves = []
    with open(file_path, "r") as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        i = 0
        with tqdm(total=100) as pbar:
            while game:
                game = chess.pgn.read_game(pgn_file)
                moves = []
                node = game
                while node.variations:
                    next_node = node.variation(0)
                    move = node.board().san(next_node.move)
                    moves.append(move)
                    node = next_node
                i += 1
                if i % 1000 == 0:
                    pbar.update(1)

    ttl_moves.append(moves)
    moves_array = np.array(ttl_moves, dtype=object)
    print(f"Size of moves_array: {moves_array.nbytes} bytes")
    del ttl_moves


if __name__ == "__main__":
    load_pgn("lichess_db_standard_rated_2013-01.pgn")
