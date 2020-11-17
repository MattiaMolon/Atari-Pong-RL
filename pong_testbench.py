import wimblepong
import gym
from wimblepong.simple_ai import SimpleAi


class PongTestbench(object):
    def __init__(self, render=False, silent=True):
        self.player1 = None
        self.player2 = None
        self.total_games = 0
        self.wins1 = 0
        self.wins2 = 0
        self.render = render
        self.env = gym.make("WimblepongVisualMultiplayer-v0")
        self.silent = silent

    def init_players(self, player1, player2=None):
        self.player1 = player1
        if player2:
            self.player2 = player2
        else:
            self.player2 = SimpleAi(self.env, player_id=2)
        self.set_names()

    def switch_sides(self):
        def switch_simple_ai(player):
            if type(player) is SimpleAi:
                player.player_id = 3 - player.player_id

        op1 = self.player1
        ow1 = self.wins1
        self.player1 = self.player2
        self.wins1 = self.wins2
        self.player2 = op1
        self.wins2 = ow1

        # Ensure SimpleAi knows where it's playing
        switch_simple_ai(self.player1)
        switch_simple_ai(self.player2)

        self.env.switch_sides()
        if not self.silent:
            print("Switching sides.")

    def play_game(self):
        self.player1.reset()
        self.player2.reset()
        obs1, obs2 = self.env.reset()
        done = False
        while not done:
            action1 = self.player1.get_action(obs1)
            action2 = self.player2.get_action(obs2)
            (obs1, obs2), (rew1, rew2), done, info = self.env.step((action1, action2))

            if self.render:
                self.env.render()

            if done:
                if rew1 > 0:
                    self.wins1 += 1
                elif rew2 > 0:
                    self.wins2 += 1
                else:
                    raise ValueError("Game finished but no one won?")
                self.total_games += 1
                # print("Game %d finished." % self.total_games)

    def run_test(self, no_games=100, switch_freq=-1):
        # Ensure the testbench is in clear state
        assert self.wins1 == 0 and self.wins2 == 0 and self.total_games == 0

        if switch_freq == -1:
            # Switch once in the middle
            switch_freq = no_games // 2
        elif switch_freq in (None, 0):
            # Don't switch sides at all
            switch_freq = no_games*2

        if not self.silent:
            print("Running test: %s vs %s." % (self.player1.get_name(), self.player2.get_name()))
        for i in range(no_games):
            self.play_game()
            if i % switch_freq == switch_freq - 1:
                self.switch_sides()

        # Ensure correct state
        assert self.wins1 + self.wins2 == self.total_games

        if not self.silent:
            print("Test results:")
            print("%s vs %s" % (self.player1.get_name(), self.player2.get_name()))
            print("%d : %d" % (self.wins1, self.wins2))
            print("-"*40)

    def set_names(self):
        def verify_name(name):
            # TODO: some ASCII/profanity checks?
            return type(name) is str and 0 < len(name) <= 26

        name1 = self.player1.get_name()
        name2 = self.player2.get_name()

        if not verify_name(name1):
            raise ValueError("Name", name1, "not correct")
        if not verify_name(name2):
            raise ValueError("Name", name2, "not correct")

        self.env.set_names(name1, name2)

    def get_agent_score(self, agent):
        if agent is self.player1:
            return self.wins1, self.total_games
        elif agent is self.player2:
            return self.wins2, self.total_games
        else:
            raise ValueError("Agent not found in the testbench!")


