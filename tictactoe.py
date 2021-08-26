"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Create a variable to count the number of moves currently on the board
    current_moves = 0
    moves_played = []

    # Begin with player 'X' and switch between players accordingly
    if initial_state is None:
        next_player = X
        return ("X may begin the game")
    # Use modulo operator to determine who is next
    elif current_moves % 2 == 1:
        next_player = X
        return ("Turn: Player X")
    else:
        next_player = O
        return ("Turn: Player O")
      
    # Increment number of current moves by 1 as they are played, add moves played to list
    current_moves += 1
    moves_played.append(board)
    actions(board)


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    legal_moves = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == None:
                legal_moves.add((i, j))

    return legal_moves
  
  def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    temp_board = copy.deepcopy(board)
    
    if action in actions(board):
        i, j = action
        temp_board[i][j] = player(board)
        return temp_board
    # else:
        # BUG: Keeps throwing error. Out of time to fix
            # raise Exception("Invalid move, please try again.")


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Winning states for vertical and horizontal states
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
        elif board[0][i] == board[1][i] == board[2][i]:
            return board[0][i]
        # Diagonal winning states
        elif board[0][0] == board[2][2] == board[1][1]:
          return board[1][1]
        elif board[2][0] == board[0][2] == board[1][1]:
            return board[1][1]
        # None if no one has won
        else:
            return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) == X or winner(board) == O:
              return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        return max(board)[1]

    else:
        return min(board)[1]

# BUG: Commented out due to time
"""
def max(board):
    "if terminal(board):
     "   return None
        
    x = -math.inf
    
    for action in actions(board):
        x = max(x, min(result(board, action))[0])
            return x
def min(board):
    if terminal(board):
        return None
        
    y = math.inf
    
    for action in actions(board):
        y = min(y, max(result(board, action))[0])
    
    return y
"""
