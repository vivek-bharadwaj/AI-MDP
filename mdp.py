import numpy as np
from copy import deepcopy


def get_north_coordinates(row, col):
    """
    Find new row and column by going North
    row - 1, col
    :param row: row
    :param col: col
    :return:
    """
    if row - 1 < 0:
        return row, col
    else:
        return row - 1, col


def get_south_coordinates(row, col):
    """
    Find new row and column by going South
    row + 1, col
    :param row: row
    :param col: col
    :return:
    """
    if row + 1 > grid_size - 1:
        return row, col
    else:
        return row + 1, col


def get_east_coordinates(row, col):
    """
    Find new row and column by going East
    row, col + 1
    :param row: row
    :param col: col
    :return:
    """
    if col + 1 > grid_size - 1:
        return row, col
    else:
        return row, col + 1


def get_west_coordinates(row, col):
    """
    Find new row and column by going West
    row, col - 1
    :param row: row
    :param col: col
    :return:
    """
    if col - 1 < 0:
        return row, col
    else:
        return row, col - 1

# 0th index -> turn left, 1st index - turn right


directions = {"UP": ["LEFT", "RIGHT"],
              "DOWN": ["RIGHT", "LEFT"],
              "RIGHT": ["UP", "DOWN"],
              "LEFT": ["DOWN", "UP"]}


def find_new_position(direction, row, col):
    if direction == "UP":
        row, col = get_north_coordinates(row, col)
    if direction == "DOWN":
        row, col = get_south_coordinates(row, col)
    if direction == "RIGHT":
        row, col = get_east_coordinates(row, col)
    if direction == "LEFT":
        row, col = get_west_coordinates(row, col)
    return [row, col]


def run_simulations(start, end, reward, policy):
    """
    Check correctness of optimal policy by simulating with random numbers in grid
    :param start: Start state [row,col] for car
    :param end: Terminal state [row, col] for car
    :param reward: initial rewards
    :param policy: initial policy
    :return: int average of 10 random seeds
    """
    cost_list = [0.0] * 10
    for j in range(10):
        pos = start
        np.random.seed(j)
        swerve = np.random.random_sample(1000000)
        k = 0
        cost_value = 0.0
        while pos[0] != end[0] or pos[1] != end[1]:
            move = policy[pos[0]][pos[1]]
            if swerve[k] > 0.7:
                if swerve[k] > 0.8:
                    if swerve[k] > 0.9:
                        # Turn Right Turn Right
                        move = directions[directions[move][1]][1]
                    else:
                        # Turn Right
                        move = directions[move][1]
                else:
                    # Turn Left
                    move = directions[move][0]
            pos = find_new_position(move, pos[0], pos[1])
            cost_value += reward[pos[0]][pos[1]]
            k += 1
        cost_list[j] = cost_value
    avg = np.float64(np.floor(sum(cost_list) / 10))
    return int(avg)


def max_p_utility_prod(u, x, y):
    """
    Calculate Maximum expected utility Sum(p*U[S1])
    :param u: Utility
    :param x: row
    :param y: col
    :return: MEU
    """
    x_n, y_n = get_north_coordinates(x, y)
    x_s, y_s = get_south_coordinates(x, y)
    x_e, y_e = get_east_coordinates(x, y)
    x_w, y_w = get_west_coordinates(x, y)

    # print "Inside Max func\n", print_2d_array(utilities)
    # v1 -> value by going north (up)
    # v2 -> value by going south (down)
    # v3 -> value by going east (right)
    # v4 -> value by going west (left)

    v1 = np.float64(0.7 * u[x_n][y_n] + 0.1 * u[x_s][y_s] + 0.1 * u[x_e][y_e] + 0.1 * u[x_w][y_w])
    v2 = np.float64(0.1 * u[x_n][y_n] + 0.7 * u[x_s][y_s] + 0.1 * u[x_e][y_e] + 0.1 * u[x_w][y_w])
    v3 = np.float64(0.1 * u[x_n][y_n] + 0.1 * u[x_s][y_s] + 0.7 * u[x_e][y_e] + 0.1 * u[x_w][y_w])
    v4 = np.float64(0.1 * u[x_n][y_n] + 0.1 * u[x_s][y_s] + 0.1 * u[x_e][y_e] + 0.7 * u[x_w][y_w])

    max_val = max(v1, v2, v3, v4)
    return max_val


def find_optimal_policy(u, p):
    """
    :param u: Utility 2d array
    :param p: Policy 2d array
    :return: Optimal Policy (str)
    """
    n = grid_size
    for i in range(n):
        for j in range(n):
            x1, y1 = get_north_coordinates(i, j)
            x2, y2 = get_south_coordinates(i, j)
            x3, y3 = get_east_coordinates(i, j)
            x4, y4 = get_west_coordinates(i, j)

            v1, v2, v3, v4 = u[x1][y1], u[x2][y2], u[x3][y3], u[x4][y4]
            max_val = max(v1, v2, v3, v4)

            if max_val == v1:
                p[i][j] = "UP"
            elif max_val == v2:
                p[i][j] = "DOWN"
            elif max_val == v3:
                p[i][j] = "RIGHT"
            elif max_val == v4:
                p[i][j] = "LEFT"
    return p


def value_iteration(utilities, rewards, policies, x_end, y_end, epsilon=0.1):
    """
        Returns new utility matrix
        max
            Up:
            row - 1, col
            0.7 * utility[row - 1, col] + 0.1 * utility[...] repeat 3 times

            Down:
            row + 1, col
            0.7 * utility[row + 1, col] + 0.1 * ...

            Left:
            row, col - 1

            Right:
            row, col + 1
    """

    gamma = 0.9

    while True:
        utilities_copy = deepcopy(utilities)
        delta = 0

        for x in range(grid_size):
            for y in range(grid_size):
                # if x,y is a terminal node do not update utility
                if x == x_end and y == y_end:
                    utilities[x][y] = rewards[x][y]
                    policies[x][y] = None
                    continue
                utilities[x][y] = rewards[x][y] + gamma * max_p_utility_prod(utilities, x, y)

                if abs(utilities_copy[x][y] - utilities[x][y]) > delta:
                    delta = abs(utilities_copy[x][y] - utilities[x][y])

        if delta <= epsilon * (1 - gamma) / gamma:
            return utilities


if __name__ == '__main__':

    input_file = open("input.txt", "r")
    output_file = open("output.txt", "w")
    grid_size = int(input_file.readline().strip())
    num_cars = int(input_file.readline().strip())
    num_obstacles = int(input_file.readline().strip())

    obstacles = [input_file.readline().strip() for _ in range(int(num_obstacles))]
    obstacles = [map(int, o.split(',')) for o in obstacles]

    cars_start = [input_file.readline().strip() for _ in range(int(num_cars))]
    cars_start = [map(int, c.split(',')) for c in cars_start]

    cars_end = [input_file.readline().strip() for _ in range(int(num_cars))]
    cars_end = [map(int, c.split(',')) for c in cars_end]

    for car in range(num_cars):
        rewards = [[-1.0 for x in range(grid_size)] for y in range(grid_size)]
        utilities = [[0.0 for x in range(grid_size)] for y in range(grid_size)]
        policies = [[None for i in range(grid_size)] for j in range(grid_size)]

        x_end, y_end = cars_end[car][0], cars_end[car][1]
        rewards[x_end][y_end] = 99

        for o in obstacles:
            x, y = o[0], o[1]
            rewards[x][y] = -101

        rewards = np.transpose(rewards)
        policies = np.transpose(policies)

        utilities = value_iteration(utilities, rewards, policies, y_end, x_end, epsilon=0.1)
        policies = find_optimal_policy(utilities, policies)

        mean_rewards = run_simulations(cars_start[car][::-1], cars_end[car][::-1], rewards, policies)
        output_file.write(str(mean_rewards) + "\n")
