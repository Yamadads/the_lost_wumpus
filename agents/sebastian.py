import random
import numpy as np
from action import Action
from numpy import unravel_index
import copy


class Agent:

    def __init__(self, p, pj, pn, height, width, areaMap):
        self.p = p
        self.pj = pj
        self.pn = pn
        self.height = height
        self.width = width
        self.map = areaMap

        self.hist = []
        self.temp_hist = []
        for y in range(self.height):
            self.hist.append([])
            for x in range(self.width):
                self.hist[y].append(1.0)
        self._normalize_hist()
        self.max_prob_threshold = self.hist[0][0]*2
        self.move_dir = {}
        self.move_dir[Action.UP] = [0,1]
        self.move_dir[Action.DOWN] = [0,-1]
        self.move_dir[Action.LEFT] = [1,0]
        self.move_dir[Action.RIGHT] = [-1,0]

        self.to_move_dir = {}
        self.to_move_dir[Action.DOWN] = [0,1]
        self.to_move_dir[Action.UP] = [0,-1]
        self.to_move_dir[Action.LEFT] = [-1,0]
        self.to_move_dir[Action.RIGHT] = [1,0]

        self.prob_param = {}
        self.prob_param['J',True] = self.pj
        self.prob_param['J',False] = 1.0 - self.pj
        self.prob_param['.',True] = self.pn
        self.prob_param['.',False] = 1.0 - self.pn
        self.prob_param['W',True] = self.pn
        self.prob_param['W',False] = 1.0 - self.pn

        self.exit_position = [-1,-1]
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] == 'W':
                    self.exit_position = [i,j]
                    break
            if self.exit_position != [-1,-1]:
                break

        self.times_moved = 0
        self.direction = Action.LEFT
        self.general_times_moved = 0
        self.target_moves = []
        return

    def _update_hist_move(self, move):
        self.temp_hist = copy.deepcopy(self.hist)
        for row in range(len(self.hist)):
            for i in range(len(self.hist[0])):
                self.hist[row][i] = self._calc_prob_in_field(row,i,move)
        self._normalize_hist()

    def _calc_prob_in_field(self, y, x, action):
        prob_sum = 0
        x += self.move_dir[action][0]
        y += self.move_dir[action][1]
        for i in self.move_dir:
            prob_sum += self._get_cyclic_temp_hist_value(y+self.move_dir[i][1],x+self.move_dir[i][0])*((1.0-self.p)/4)
        prob_sum += self._get_cyclic_temp_hist_value(y,x)*self.p
        return prob_sum

    def _get_cyclic_temp_hist_value(self, y, x):
        return self.temp_hist[y%self.height][x%self.width]

    def _update_hist_sense(self, sensor_result):
        for row in xrange(len(self.hist)):
            for i in range(len(self.hist[0])):
                self.hist[row][i] *= self.prob_param[self.map[row][i],sensor_result]
        self._normalize_hist()

    def _normalize_hist(self):
        hist_sum = float(np.sum(self.hist))
        for row in range(len(self.hist)):
            for i in range(len(self.hist[0])):
                self.hist[row][i] /= hist_sum

    def sense(self, sensor):
        self._update_hist_sense(sensor)
        return dir

    def move(self):
        self.general_times_moved += 1
        a = np.array(self.hist)
        max_prob_idx = unravel_index(a.argmax(), a.shape)
        max_prob = self.hist[max_prob_idx[0]][max_prob_idx[1]]
        if (max_prob) > self.max_prob_threshold and self.general_times_moved>10:
            dir = self._get_best_move(max_prob_idx)
            #self._check_move(dir,max_prob_idx)
        else:
            dir = self._snake_move()
        self._update_hist_move(dir)
        return dir

    def _get_best_move(self, actual_position):
        actual_x = actual_position[1]
        actual_y = actual_position[0]
        target_x = self.exit_position[1]
        target_y = self.exit_position[0]
        z_x = target_x-actual_x
        z_y = target_y-actual_y

        if z_x > 0:
            if z_x > (0.5*self.width):
                return Action.LEFT
            else:
                return Action.RIGHT
        elif z_x < 0:
            if abs(z_x) > (0.5*self.width):
                return Action.RIGHT
            else:
                return Action.LEFT

        if z_y > 0:
            if z_y > (0.5*self.height):
                return Action.UP
            else:
                return Action.DOWN
        elif z_y <0:
            if abs(z_y) > (0.5*self.height):
                return Action.DOWN
            else:
                return Action.UP

        self.hist[actual_position[0]][actual_position[1]] *= 0.5
        self._normalize_hist()

        return random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])


    def _snake_move(self):
        if self.times_moved < 0.5*self.width - 1:
            self.times_moved += 1
            return self.direction
        else:
            self.times_moved = 0
            self.direction = Action.RIGHT if self.direction == Action.LEFT else Action.LEFT
            return Action.DOWN

    def histogram(self):
        #print(self.hist)
        return self.hist
