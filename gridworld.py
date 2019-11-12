import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        self.shape = [13,13]
        self.directions = [np.array((-1,0)), np.array((0,1)), np.array((1,0)), np.array((0,-1))]
        self.tostate = np.load('tostate.npy').item()
        self.tocell = {v:k for k,v in self.tostate.items()}        
        nS = np.prod(shape)
        nS = 104
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}        
        self.grid = np.load('layout.npy')
        self.grid[1,1] = 2
        
        
        for i in range(1,13):
            for j in range(1,13):              
                
                self.currentcell = tuple([i,j])  

                if self.currentcell in self.tostate.keys():

                    s = self.tostate[self.currentcell] 
                    P[s] = {a : [] for a in range(nA)}

                    nextcell = tuple(self.currentcell + self.directions[UP])                    
                    if nextcell not in self.tostate.keys():
                        P[s][UP] = [(1.0,s,0,False)]
                    else:
                        #print('in',state,self.currentcell)
                        state = self.tostate[nextcell]
                        P[s][UP] = [(1.0,state,0,False)]

                    nextcell = tuple(self.currentcell + self.directions[DOWN])                    
                    if nextcell not in self.tostate.keys():
                        P[s][DOWN] = [(1.0,s,0,False)]
                    else:
                        state = self.tostate[nextcell]
                        P[s][DOWN] = [(1.0,state,0,False)]

                    nextcell = tuple(self.currentcell + self.directions[RIGHT])
                    if nextcell not in self.tostate.keys():
                        P[s][RIGHT] = [(1.0,s,0,False)]
                    else:
                        state = self.tostate[nextcell]
                        P[s][RIGHT] = [(1.0,state,0,False)]

                    nextcell = tuple(self.currentcell + self.directions[LEFT])
                    if nextcell not in self.tostate.keys():
                        P[s][LEFT] = [(1.0,s,0,False)]
                    else:
                        state = self.tostate[nextcell]
                        P[s][LEFT] =[(1.0,state,0,False)]

        
        P[0][UP] = [(1.0,0,1.0,True)]
        P[0][DOWN] = [(1.0,0,1.0,True)]
        P[0][RIGHT] = [(1.0,0,1.0,True)]
        P[0][LEFT] = [(1.0,0,1.0,True)]
        np.save("P.npy",P)
        # it = np.nditer(grid, flags=['multi_index'])

        # while not it.finished:
        #     s = it.iterindex
        #     print(s)
        #     y, x = it.multi_index

        #     # P[s][a] = (prob, next_state, reward, is_done)
        #     P[s] = {a : [] for a in range(nA)}

        #     is_done = lambda s: s == 0 or s == (nS - 1)
        #     reward = 0.0 if is_done(s) else -1.0

        #     # We're stuck in a terminal state
        #     if is_done(s):
        #         P[s][UP] = [(1.0, s, reward, True)]
        #         P[s][RIGHT] = [(1.0, s, reward, True)]
        #         P[s][DOWN] = [(1.0, s, reward, True)]
        #         P[s][LEFT] = [(1.0, s, reward, True)]
        #     # Not a terminal state
        #     else:
        #         ns_up = s if y == 0 else s - MAX_X
        #         ns_right = s if x == (MAX_X - 1) else s + 1
        #         ns_down = s if y == (MAX_Y - 1) else s + MAX_X
        #         ns_left = s if x == 0 else s - 1
        #         P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
        #         P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
        #         P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
        #         P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

        #     it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout

         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
