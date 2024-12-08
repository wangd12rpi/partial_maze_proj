if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_simulations = [500, 1000, 2000, 3000, 4000]
    time_per_run = [12.56, 16.94, 31.47, 46.02, 59.74]

    plt.figure(figsize=(8, 6))
    plt.plot(num_simulations, time_per_run, marker='o', linestyle='-', color='b')

    plt.xlabel('Number of Simulations', fontsize=12)
    plt.ylabel('Time per Run (seconds)', fontsize=12)
    plt.title('MCTS Performance in 9x9 Mazes: Simulations vs. Time', fontsize=14)

    plt.grid(True)

    plt.show()


    depth = [11, 12, 13, 14]
    time_per_run = [4.30, 12.10, 35.25, 393.33]

    plt.figure(figsize=(8, 6))
    plt.plot(depth, time_per_run, marker='o', linestyle='-', color='r')

    plt.xlabel('Depth', fontsize=12)
    plt.ylabel('Time per Run (seconds)', fontsize=12)
    plt.title('Forward Search Performance in 9x9 Mazes: Depth vs. Time', fontsize=14)

    plt.grid(True)


    plt.show()