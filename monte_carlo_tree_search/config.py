class Config:
    levels = 3
    numIters = 20
    numEps = 15  # num self-play games to simulate during a new iteration.
    probabilisticThreshold = 15  # essentially useless if it is > level
    updateThreshold = 0.55  # model replacement win threshold
    maxlenOfQueue = 2000  # num game examples to train network
    numMCTSSims = 200  # num moves for MCTS to simulate
    gymCompare = 12  # num games between old and new model
    cpuct = 1
    checkpoint = "./temp/"
