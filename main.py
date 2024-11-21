import chess
from pprint import pprint


class ChessHelper:

    CHESS_PIECES = ["p", "n", "b", "q", "k", "r"]

    def __init__(self):
        self.board = chess.Board()

    def getBoard(self, debug=False):
        board_2d = []
        for rank in chess.RANK_NAMES:
            row = []
            for file in chess.FILE_NAMES:
                square = chess.square(
                    chess.FILE_NAMES.index(file), chess.RANK_NAMES.index(rank)
                )
                row.append(str(self.board.piece_at(square) or "."))
            board_2d.append(row)

        pprint(board_2d) if debug else None
        return board_2d

    def boardToMatrix(self, debug=False):
        board_2d = []
        master_board = []

        for i in range(1, 4):
            for piece in self.CHESS_PIECES:
                BIGrow = []
                for rank in chess.RANK_NAMES:
                    SMALLrow = []
                    for file in chess.FILE_NAMES:
                        square = chess.square(
                            chess.FILE_NAMES.index(file), chess.RANK_NAMES.index(rank)
                        )

                        piece_at_square = self.board.piece_at(square)

                        if i == 1:
                            SMALLrow.append(
                                piece_at_square.symbol()
                                if piece_at_square
                                and piece_at_square.symbol() == piece.lower()
                                else "."
                            )
                        elif i == 2:
                            SMALLrow.append(
                                piece_at_square.symbol()
                                if piece_at_square
                                and piece_at_square.symbol() == piece.upper()
                                else "."
                            )
                        elif i == 3:
                            SMALLrow.append(
                                piece_at_square.symbol()
                                if piece_at_square
                                and piece_at_square.symbol().upper() == piece.upper()
                                else "."
                            )

                    BIGrow.append(SMALLrow)
                master_board.append(BIGrow)

        pprint(master_board) if debug else None
        print(len(master_board))

        return master_board


def main() -> None:
    chessHelperObj = ChessHelper()
    chessHelperObj.boardToMatrix(debug=True)


if __name__ == "__main__":
    main()
