import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
import random

from sehec.envs.envcore import Environment
import sehec.models.TEM as model


class TEMenv(Environment):
    def __init__(self, environment_name="TEMenv", **env_kwargs):
        super().__init__(environment_name, **env_kwargs)
        self.metadata = {"env_kwargs": env_kwargs}
        self.pars = env_kwargs

        self.last_states = None
        self.last_states2 = None
        self.reset()

    def reset(self):
        self.global_steps = 0
        self.history = []
        self.state = [0, 0]
        self.state = np.array(self.state)
        self.state1 = [0, 0]
        self.state1 = np.array(self.state1)
        self.state2 = [0, 0]
        self.state2 = np.array(self.state2)
        self.last_states = np.empty(shape=(self.pars['batch_size'], 1))
        self.last_states[:] = np.nan
        self.last_states1 = np.empty(shape=(self.pars['batch_size'], 2))
        self.last_states1[:] = np.nan
        self.last_states2 = np.empty(shape=(self.pars['batch_size'], 2))
        self.last_states2[:] = np.nan
        observation = self.state

        return observation, self.state

    def make_observation(self):
        return self.state

    def make_environment(self):
        n_envs = len(self.pars['widths'])
        adjs, trans = [], []

        for env in range(n_envs):

            width = self.pars['widths'][env]

            if self.pars['world_type'] == 'square':
                adj, tran = self.square_world(width, self.pars['stay_still'])
            else:
                raise ValueError('incorrect world specified')

            adjs.append(adj)
            trans.append(tran)

        return adjs, trans

    def square_world(self, width, stay_still):
        states = int(width ** 2)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if stay_still:
                adj[i, i] = 1
            # up - down
            if i + width < states:
                adj[i, i + width] = 1
                adj[i + width, i] = 1
                # left - right
            if np.mod(i, width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        return adj, tran

    def step(self, policy, index, n_walk):
        # Step through environment depending on given action policy
        self.global_steps += 1
        observations = np.zeros(shape=(self.pars['batch_size'], 1, self.pars['t_episode'] + 1))
        observations1 = np.zeros(shape=(self.pars['batch_size'], 2, self.pars['t_episode'] + 1))
        # new_states = np.zeros(shape=(self.pars['batch_size'], 2, self.pars['t_episode'] + 1))
        rewards = []

        actions1 = np.zeros((self.pars['batch_size'], 2, self.pars['t_episode']))
        direcs = np.zeros(shape=(self.pars['batch_size'], 4, self.pars['t_episode']))
        direcs1 = np.zeros(shape=(self.pars['batch_size'], 4, self.pars['t_episode']))

        adjs, trans = self.make_environment()

        for batch in range(self.pars['batch_size']):
            room_width = self.pars['widths'][batch]
            room_depth = self.pars['widths'][batch]
            batch_history = []
            if np.isnan(self.last_states[batch]).any():
                self.state1 = [round(random.uniform(-room_width / 2, room_width / 2)),
                               round(random.uniform(-room_depth / 2, room_depth / 2))]
                allowed_states = np.where(np.sum(adjs[batch], 1) > 0)[0]
                self.state = int(np.random.choice(allowed_states))
            else:
                self.state = int(self.last_states[batch])
                self.state1 = self.last_states1[batch]

            observations[batch, 0, 0] = self.state
            observations1[batch, :, 0] = self.state1
            # new_states[batch, :, 0] = self.state
            for step in range(self.pars['t_episode']):
                available = np.where(trans[batch][int(observations[batch, 0, step]), :] > 0)[0].astype(int)
                new_state = np.random.choice(available)

                if adjs[batch][int(observations[batch, 0, step]), new_state] == 1:
                    observations[batch, 0, step + 1] = new_state
                else:
                    observations[batch, 0, step + 1] = int(cp.deepcopy(observations[0, step]))

                prev_dir, _ = rectangle_relation(observations[batch, 0, step], observations[batch, 0, step + 1],
                                                 room_width, room_depth)
                if prev_dir < 4:
                    direcs[batch, prev_dir, step] = 1
                # Generate action from given policy
                action1, direc1 = policy()
                actions1[batch, :, step] = action1
                direcs1[batch, :, step] = direc1

                # Determine state transitioned to
                new_state1 = [self.state1[n] + self.pars['agent_step_size'] * i for n, i in enumerate(action1)]
                # new_state1 = np.array([np.clip(new_state1[0], a_min=-room_width / 2, a_max=room_width / 2),
                #                       np.clip(new_state1[1], a_min=-room_depth / 2, a_max=room_depth / 2)])
                while any(new_state1 > room_width/2) or any(new_state1 < -room_width/2):
                    action1, direc1 = policy()
                    actions1[batch, :, step] = action1
                    direcs1[batch, :, step] = direc1
                    new_state1 = [self.state1[n] + self.pars['agent_step_size'] * i for n, i in enumerate(action1)]

                reward = 0  # If you get reward, it should be coded here
                transition = {"action": action1, "state": self.state1, "next_state": new_state1,
                              "reward": reward, "step": self.global_steps}
                batch_history.append(transition)
                self.state = new_state
                self.state1 = new_state1
                observation = self.make_observation()

                observations1[batch, :, step + 1] = self.state1
                # new_states[batch, :, step + 1] = new_state
                rewards.append(reward)

            if index == n_walk - 1:
                self.last_states[batch] = np.nan
                self.last_states1[batch] = np.nan
            else:
                self.last_states[batch] = self.state
                self.last_states1[batch] = self.state1

            self.history.append(batch_history)

        return observations, rewards, actions, direcs, observations1, direcs1

    def plot_trajectory(self, history_data=None, ax=None):
        if history_data is None:
            history_data = self.history
            # history_data = []
            # for i in range(0, 400, 25):
            #     history_data.append(self.history[i:i+25])
        if ax is None:
            mlp.rc('font', size=6)
            fig = plt.figure(figsize=(8, 6))

        for batch in range(16):
            ax = plt.subplot(4, 4, batch + 1)
            room_width = self.pars['widths'][batch]
            room_depth = room_width

            ax.plot([-room_width / 2, room_width / 2],
                    [-room_depth / 2, -room_depth / 2], "r", lw=2)
            ax.plot([-room_width / 2, room_width / 2],
                    [room_depth / 2, room_depth / 2], "r", lw=2)
            ax.plot([-room_width / 2, -room_width / 2],
                    [-room_depth / 2, room_depth / 2], "r", lw=2)
            ax.plot([room_width / 2, room_width / 2],
                    [-room_depth / 2, room_depth / 2], "r", lw=2)

            state_history = [s["state"] for s in history_data[batch]]
            next_state_history = [s["next_state"] for s in history_data[batch]]

            cmap = mlp.cm.get_cmap("plasma")
            norm = plt.Normalize(0, len(state_history))

            aux_x = []
            aux_y = []
            for i, s in enumerate(state_history):
                x_ = [s[0], next_state_history[i][0]]
                y_ = [s[1], next_state_history[i][1]]
                aux_x.append(s[0])
                aux_y.append(s[1])
                ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6)

            sc = ax.scatter(aux_x, aux_y, c=np.arange(len(state_history)),
                            vmin=0, vmax=len(state_history), cmap="plasma", alpha=0.6)
            ax.set_xlim([-6, 6])
            ax.set_xticks([-5, 0, 5])
            ax.set_ylim([-6, 6])
            ax.set_yticks([-5, 0, 5])

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(sc, cax=cbar_ax)
        cbar_ax.set_ylabel('N steps', rotation=270)
        self.history = []

        return ax


def rectangle_relation(s1, s2, width, height):
    # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
    diff = s2 - s1
    if diff == width or diff == -width * (height - 1):  # down
        direc = 0
        name = 'down'
    elif diff == -width or diff == width * (height - 1):  # up
        direc = 1
        name = 'up'
    elif diff == -1 or diff == (width - 1):  # left
        direc = 2
        name = 'left'
    elif diff == 1 or diff == -(width - 1):  # right
        direc = 3
        name = 'right'
    elif diff == 0:
        direc = 4
        name = 'stay still'
    else:
        raise ValueError('impossible action')

    return direc, name


def actions(direc):
    if direc[0] == 1:
        action = [-1, 0]
    elif direc[1] == 1:
        action = [1, 0]
    elif direc[2] == 1:
        action = [0, -1]
    elif direc[3] == 1:
        action = [0, 1]
    else:
        action = [0, 0]

    return action

def direction(action):
    # Turns action [x,y] into direction [R,L,U,D]
    x, y = action
    direc = np.zeros(shape=4)
    if x > 0 and y == 0:
        d = 0
        name = 'right'
        direc[d] = 1
    elif x < 0 and y == 0:
        d = 1
        name = 'left'
        direc[d] = 1
    elif x == 0 and y > 0:
        d = 2
        name = 'up'
        direc[d] = 1
    elif x == 0 and y < 0:
        d = 3
        name = 'down'
        direc[d] = 1
    else:
        ValueError('impossible action')

    return direc