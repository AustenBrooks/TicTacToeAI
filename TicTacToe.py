import numpy as np
from random import randrange


class TicTacToe:
    symbol = {-1: ' ', 0: 'O', 1: 'X'}
    board = []
    NUM_STATES = 19683  # there are 3^9 possible states, some (ex: all X's) will never be obtained
    ACTIONS = 9
    encodeBoard = {}

    def __init__(self):
        for i in range(9):
            self.board.append(' ')
        self.setupDict()

    def setupDict(self):
        MARKERS = [' ', 'O', 'X']
        state = 0
        for a in MARKERS:
            for b in MARKERS:
                for c in MARKERS:
                    for d in MARKERS:
                        for e in MARKERS:
                            for f in MARKERS:
                                for g in MARKERS:
                                    for h in MARKERS:
                                        for i in MARKERS:
                                            self.encodeBoard[str([a, b, c, d, e, f, g, h, i])] = state
                                            state += 1

    def render(self):
        for row in range(3):
            print(self.board[row*3] + ' | ' + self.board[row*3 + 1] + ' | ' + self.board[row*3 + 2])
            if row != 2:
                print("---------")
        print()

    def checkWinner(self, player):
        # check horiz 3 in a row
        for row in range(3):
            if self.board[row * 3] != ' ':
                if self.board[row * 3] == self.board[row * 3 + 1] and self.board[row * 3] == self.board[row * 3 + 2]:
                    if self.board[row * 3] == self.symbol[player]:
                        return 1
                    else:
                        return -1
        # check vert 3 in a row
        for col in range(3):
            if self.board[col] != ' ':
                if self.board[col] == self.board[col + 3] and self.board[col] == self.board[col + 6]:
                    if self.board[col] == self.symbol[player]:
                        return 1
                    else:
                        return -1
        # check diagonals
        if self.board[0] != ' ':
            if self.board[0] == self.board[4] and self.board[0] == self.board[8]:
                if self.board[0] == self.symbol[player]:
                    return 1
                else:
                    return -1
        if self.board[2] != ' ':
            if self.board[2] == self.board[4] and self.board[2] == self.board[6]:
                if self.board[2] == self.symbol[player]:
                    return 1
                else:
                    return -1
        return 0

    def mark(self, tile, player):
        if player == 0:
            self.board[tile] = 'O'
        elif player == 1:
            self.board[tile] = 'X'

        nextState = self.encodeBoard[str(self.board)]
        reward = self.checkWinner(player)

        isFull = True
        for a in self.board:
            if a == ' ':
                isFull = False
        isDone = (reward != 0 or isFull)

        return nextState, reward, isDone

    def isAvailable(self, tile):
        if self.board[tile] == ' ':
            return True
        return False

    def reset(self):
        self.board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
        return 0

    def getBoard(self):
        return self.board


def boardEval(board):
    # check horiz 3 in a row
    for row in range(3):
        if board[row * 3] != ' ':
            if board[row * 3] == board[row * 3 + 1] and board[row * 3] == board[row * 3 + 2]:
                if board[row * 3] == 'O':
                    return 1
                else:
                    return -1
    # check vert 3 in a row
    for col in range(3):
        if board[col] != ' ':
            if board[col] == board[col + 3] and board[col] == board[col + 6]:
                if board[col] == 'O':
                    return 1
                else:
                    return -1
    # check diagonals
    if board[0] != ' ':
        if board[0] == board[4] and board[0] == board[8]:
            if board[0] == 'O':
                return 1
            else:
                return -1
    if board[2] != ' ':
        if board[2] == board[4] and board[2] == board[6]:
            if board[2] == 'O':
                return 1
            else:
                return -1
    return 0


def evalMove(board, isMax, depth):
    # returns the value of a move, assuming the opponent makes the best choices

    score = boardEval(board)
    # if no more moves can be made, return score
    if depth == 0:
        return score

    # if there is a winner, return the score, better scores are given to faster solutions
    if score != 0:
        return score * depth

    symbol = {True: 'O', False: 'X'}
    val = -10
    if not isMax:
        val *= -1

    # run recursive minimax algorithm (find best choice assuming opponent makes best choice)
    # run through all board tiles
    for row in range(3):
        for col in range(3):
            # find empty tiles
            if board[row * 3 + col] == ' ':
                # make move
                board[row * 3 + col] = symbol[isMax]

                # find best move recursively
                if isMax:
                    val = max(val, evalMove(board, not isMax, depth - 1))
                else:
                    val = min(val, evalMove(board, not isMax, depth - 1))

                # undo move
                board[row * 3 + col] = " "
    return val


def findBestMove(board, player, depth):
    symbol = {0: 'O', 1: 'X'}

    isMax = True
    if player == 1:
        isMax = False

    bestVal = -10
    if isMax == False:
        bestVal *= -1
    bestMove = [-1, -1]

    # run through all board tiles
    for row in range(3):
        for col in range(3):
            # find empty tiles
            if board[row * 3 + col] == ' ':
                # make move
                board[row * 3 + col] = symbol[player]

                #get value of that move
                currentVal = evalMove(board, not isMax, depth - 1)

                # undo move
                board[row * 3 + col] = " "

                #update best move
                if isMax:
                    if currentVal > bestVal:
                        bestVal = currentVal
                        bestMove = [row, col]
                else:
                    if currentVal < bestVal:
                        bestVal = currentVal
                        bestMove = [row, col]
    return 3 * bestMove[0] + bestMove[1]


env = TicTacToe()
STATES = env.NUM_STATES
ACTIONS = env.ACTIONS

qMatrix = np.zeros((STATES, ACTIONS))
# constants for training model
EVOLUTIONS = 5000
STEPS_PER_RUN = 6  # should never come into play, but might stop game if somehow game didnt stop once the board was full
LEARNING_RATE = 0.75
GAMMA = 0.95

epsilon = 0.9

SHOW_EVOLUTION = 500  # renders the board every X amount of evolutions

gamesWon, gamesLost, gamesTied = 0, 0, 0
recentGamesWon, recentGamesTied, recentGamesLost = 0, 0, 0

# train model
rewards = []
for episode in range(EVOLUTIONS):
    currentState = env.reset()
    if episode % SHOW_EVOLUTION == 0:
        env.render()
    for temp in range(STEPS_PER_RUN):
        if np.random.uniform(0, 1) > epsilon:
            actionArray = np.argsort(qMatrix[currentState, :])
            action = actionArray[-1]
            actionIndex = -1
            while not env.isAvailable(action):
                action = actionArray[actionIndex]
                actionIndex -= 1
        else:
            action = randrange(0, ACTIONS)
            while not env.isAvailable(action):
                action = randrange(0, ACTIONS)

        nextState, reward, isDone = env.mark(action, 0)

        # Update q values with following formula
        qMatrix[currentState, action] = qMatrix[currentState, action] + LEARNING_RATE * (reward + GAMMA * np.max(qMatrix[nextState, :]) - qMatrix[currentState, action])

        if episode % SHOW_EVOLUTION == 0:
            env.render()

        if isDone:
            rewards.append(reward)
            epsilon -= 0.001
            if epsilon < 0.01:
                epsilon = 0.01
            break

        board = env.getBoard()
        depth = 0
        for tile in board:
            if tile == ' ':
                depth += 1
        # take a random action 10% of the time so the AI can get experience in non optimal boards
        if np.random.uniform(0, 1) > 0.90:
            enemy = randrange(0, ACTIONS)
            while not env.isAvailable(enemy):
                enemy = randrange(0, ACTIONS)
        else:
            enemy = findBestMove(board, 1, depth)

        enemyNextState, enemyReward, isDone = env.mark(int(enemy), 1)

        qMatrix[currentState, action] = qMatrix[currentState, action] + LEARNING_RATE * (enemyReward * -1 + GAMMA * np.max(qMatrix[nextState, :]) - qMatrix[currentState, action])

        currentState = enemyNextState

        if episode % SHOW_EVOLUTION == 0:
            env.render()
        if isDone:
            rewards.append(enemyReward * -1)
            break

    if reward == 1:
        gamesWon += 1
    if reward == 0 and enemyReward == 0:
        gamesTied += 1
    if enemyReward == 1:
        gamesLost += 1
    if episode > (EVOLUTIONS - 101):
        if reward == 1:
            recentGamesWon += 1
        if reward == 0 and enemyReward == 0:
            recentGamesTied += 1
        if enemyReward == 1:
            recentGamesLost += 1

print("The AI played ", EVOLUTIONS, " games, here are the overall stats:")
print("Games won: ", gamesWon, "Games Lost: ", gamesLost, "Games Tied: ", gamesTied)
print("These are the stats of the 100 most recent games played:")
print("Games won: ", recentGamesWon, "Games Lost: ", recentGamesLost, "Games Tied: ", recentGamesTied)

# test run

while True:
    currentState = env.reset()
    env.render()
    for temp in range(STEPS_PER_RUN):
        actionArray = np.argsort(qMatrix[currentState, :])
        action = actionArray[-1]
        actionIndex = -1
        while not env.isAvailable(action):
            action = actionArray[actionIndex]
            actionIndex -= 1

        nextState, reward, isDone = env.mark(action, 0)

        currentState = nextState
        env.render()

        if isDone:
            break

        while True:
            enemy = int(input("Enter the tile you want to choose (0-8):"))
            if not(0 <= enemy < 9):
                print("Invalid input")
            elif env.isAvailable(int(enemy)):
                break

        nextState, garbage, isDone = env.mark(int(enemy), 1)
        currentState = nextState

        env.render()
        if isDone:
            break
    input("Press enter to continue")
