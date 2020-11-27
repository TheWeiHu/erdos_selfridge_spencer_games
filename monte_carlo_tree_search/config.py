class Config:
    levels = 10
    numIters = 50
    numEps = 50  # num self-play games to simulate during a new iteration.
    probabilisticThreshold = 15  # essentially useless if it is > level
    updateThreshold = 0.55  # model replacement win threshold
    maxlenOfQueue = 2000  # num game examples to train network
    numMCTSSims = 200  # num moves for MCTS to simulate
    gymCompare = 30  # num games between old and new model
    cpuct = 1
    maxIterHistory = 10 # only train using examples from the last X iterations
    checkpoint = "./temp/"
    suboptimality = 0.25  # percentage change of the suboptimal agent playing randomly.
