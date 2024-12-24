import random
import matplotlib.pyplot as plt
import time

''' 
This program allows for Basic CounterFactual Regret Minimisation 
for ROCK, PAPER, Scissors, Lizard, Spock. This Program is mostly based 
on the Paper by Todd W.Neller and MarcLanctot. There are also functions
based on code by Pranav Ahluwalia. Links will be below.

https://www.ma.imperial.ac.uk/~dturaev/neller-lanctot.pdf
https://www.pranav.ai/CFRM-RPS
'''

# Rock Beats Lizard and Scissors and loses to Paper and Spock
# Paper Beats Rock and Spock and loses to Scissors and Lizard
# Scissors Beats Paper and Lizard and loses to Rock and Spock
# Lizard Beats Paper and Spock and loses to Rock and Scissors
# Spock Beats Rock and Scissors and loses to Paper and Lizard

# This is the main class for the trainer.  RPSLSP = Rock, Paper, Scissors, Lizard, Spock
class rpslspTrainer:
    def __init__(self, opp_strategy):
        self.ROCK = 0
        self.PAPER = 1
        self.SCISSORS = 2
        self.LIZARD = 3
        self.SPOCK = 4
        self.NUM_ACTIONS = 5
        
        # Initialising player arrays
        self.regret_sum = [0,0,0,0,0]
        self.strategy = [0,0,0,0,0]
        self.strategy_sum = [0,0,0,0,0]

        # Arrays to keep track of strategies at certain iterations
        self.rockstrats = []
        self.paperstrats = []
        self.scissorsstrats = []
        self.lizardstrats = []
        self.spockstrats = []

        # Initialising opposition arrays
        self.opp_strategy = opp_strategy
        self.opp_regret_sum = [0,0,0,0,0]
        self.opp_strategy_sum = [0,0,0,0,0]

    # Gets the current strategy for the player. 
    # Returns an array for the strategy and the sum of strategies
    # Strategy array follows structure like this [0.4, 0.3, 0.1] as the probabilities of playing Rock, Paper, Scissors respectively
    def get_strategy(self):
        normalising_sum = 0
        self.strategy = [0,0,0,0,0]
        
        # If regret less than or equal to 0, that option is not factored into the strategy
        for x in range(self.NUM_ACTIONS):
            if self.regret_sum[x] > 0:
                self.strategy[x] = self.regret_sum[x]
            else :
                self.strategy[x] = 0
            normalising_sum += self.strategy[x]
        
        # Normalises strategy probabilities so they add to 1
        # Initial strategy is split evenly between RPS
        for x in range(self.NUM_ACTIONS):
            if normalising_sum > 0:
                self.strategy[x] = self.strategy[x] / normalising_sum
            else :
                self.strategy[x] = 1.0 / self.NUM_ACTIONS
            self.strategy_sum[x] += self.strategy[x]
        
        
        return self.strategy, self.strategy_sum
    
    # Same as player get_strategy() but for the opponent
    def get_strategy_opp(self):
        normalising_sum = 0
        self.opp_strategy = [0,0,0,0,0]
        
        for x in range(self.NUM_ACTIONS):
            if self.opp_regret_sum[x] > 0:
                self.opp_strategy[x] = self.opp_regret_sum[x]
            else :
                self.opp_strategy[x] = 0
            normalising_sum += self.opp_strategy[x]
        
        for x in range(self.NUM_ACTIONS):
            if normalising_sum > 0:
                self.opp_strategy[x] = self.opp_strategy[x] / normalising_sum
            else :
                self.opp_strategy[x] = 1.0 / self.NUM_ACTIONS
            self.opp_strategy_sum[x] += self.opp_strategy[x]
        
        
        return self.opp_strategy, self.opp_strategy_sum

    # Gets an action based on the probabilities of the player strategy
    def get_action(self, strategy):
        r = random.uniform(0,1)
        a = 0
        cumulative_probability = 0

        while ( a < self.NUM_ACTIONS -1):
            cumulative_probability += strategy[a]
            if r < cumulative_probability:
                break
            a += 1
        return a

    # Training algorithm based on https://www.pranav.ai/CFRM-RPS
    def train(self, iterations):
        iteration = 0
        action_utility = [0,0,0,0,0]
        actions = 5
        for i in range(0,iterations):
            avg = self.get_avg_strategy()

            # Keep track of the strategies every 100 iterations (for graph production)
            if iteration & 100 == 0:
                self.rockstrats.append(avg[0])
                self.paperstrats.append(avg[1])
                self.scissorsstrats.append(avg[2])
                self.lizardstrats.append(avg[3])
                self.spockstrats.append(avg[4])

            # Retrieve Actions
            t = self.get_strategy()
            strategy = t[0]
            strategySum = t[1]

            my_action = self.get_action(strategy)
            # Define an arbitrary opponent strategy from which to adjust
            other_action = self.get_action(self.opp_strategy)  

            # Opponent Chooses Rock
            if other_action == 0:
                # Utility(Paper) = 1
                action_utility[1] = 1
                # Utility(Scissors) = -1
                action_utility[2] = -1

                # Utility(Lizard) = -1
                action_utility[3] = -1
                # Utility(Spock) = 1
                action_utility[4] = 1

            # Opopnent Chooses Paper
            elif other_action == 1:
                # Utility(Rock) = -1
                action_utility[0] = -1
                # Utility(Scissors) = 1
                action_utility[2] = 1

                # Utility(Lizard) = 1
                action_utility[3] = 1
                # Utility(Spock) = -1
                action_utility[4] = -1

            # Opponent Chooses scissors
            elif other_action == 2:
                # Utility(Rock) = 1
                action_utility[0] = 1
                # Utility(Paper) = -1
                action_utility[1] = -1

                # Utility(Lizard) = -1
                action_utility[3] = -1
                # Utility(Spock) = 1
                action_utility[4] = 1
            

            # Opopnent Chooses Lizard
            elif other_action == 3:
                # Utility(Rock) = 1
                action_utility[0] = 1
                # Utility(Paper) = -1
                action_utility[1] = -1
                # Utility(Scissors) = 1
                action_utility[2] = 1
                
                # Utility(Spock) = -1
                action_utility[4] = -1
                
            # Opopnent Chooses Spock
            elif other_action == 4:
                # Utility(Rock) = 1
                action_utility[0] = -1
                # Utility(Scissors) = 1
                action_utility[2] = -1
                # Utility(Paper) = 1
                action_utility[1] = 1

                # Utility(Lizard) = 1
                action_utility[3] = 1
                


            # Add the regrets from this decision
            for i in range(self.NUM_ACTIONS):
                self.regret_sum[i] += action_utility[i] - action_utility[my_action] + 0.1
            iteration +=1
    
    # Nash equilibrium based on https://www.pranav.ai/CFRM-RPS
    def nash_equilibrium(self, iterations):
        iteration = 0
        action_utility = [0,0,0,0,0]
        strategy_sum1 = [0,0,0,0,0]
        strategy_sum2 = [0,0,0,0,0]

        regret_sum1 = [0,0,0,0,0]
        regret_sum2 = [0,0,0,0,0]

        for x in range(iterations):
            
            t1 = self.get_strategy()
            strategy1 = t1[0]
            strategy_sum1 = t1[1]
            my_action = self.get_action(strategy1)

            t2 = self.get_strategy_opp()
            strategy2 = t2[0]
            strategy_sum2 = t2[1]
            other_action = self.get_action(strategy2)

            # Opponent Chooses Rock
            if other_action == 0:
                # Utility(Paper) = 1
                action_utility[1] = 1
                # Utility(Scissors) = -1
                action_utility[2] = -1

                # Utility(Lizard) = -1
                action_utility[3] = -1
                # Utility(Spock) = 1
                action_utility[4] = 1

            # Opopnent Chooses Paper
            elif other_action == 1:
                # Utility(Rock) = -1
                action_utility[0] = -1
                # Utility(Scissors) = 1
                action_utility[2] = 1

                # Utility(Lizard) = 1
                action_utility[3] = 1
                # Utility(Spock) = -1
                action_utility[4] = -1

            # Opponent Chooses scissors
            elif other_action == 2:
                # Utility(Rock) = 1
                action_utility[0] = 1
                # Utility(Paper) = -1
                action_utility[1] = -1

                # Utility(Lizard) = -1
                action_utility[3] = -1
                # Utility(Spock) = 1
                action_utility[4] = 1
            

            # Opopnent Chooses Lizard
            elif other_action == 3:
                # Utility(Rock) = 1
                action_utility[0] = 1
                # Utility(Paper) = -1
                action_utility[1] = -1
                # Utility(Scissors) = 1
                action_utility[2] = 1
                
                # Utility(Spock) = -1
                action_utility[4] = -1
                
            # Opopnent Chooses Spock
            elif other_action == 4:
                # Utility(Rock) = 1
                action_utility[0] = -1
                # Utility(Scissors) = 1
                action_utility[2] = -1
                # Utility(Paper) = 1
                action_utility[1] = 1

                # Utility(Lizard) = 1
                action_utility[3] = 1
            

            # Add the regrets from this decision
            for i in range(0,self.NUM_ACTIONS):
                self.regret_sum[i] += action_utility[i] - action_utility[my_action]
                self.opp_regret_sum[i] += -(action_utility[i] - action_utility[my_action])
            # print("self", self.strategy_sum)
            # print("opp", self.opp_strategy_sum)
        return strategy_sum1, strategy_sum2

    # Compares players actions to opponents and change BOTH strategies accordingly
    # Returns player strategy and opponent strategy
    def rps_to_nash(self, iterations):
        strats = self.nash_equilibrium(iterations)
        s1 = sum(strats[0])
        s2 = sum(strats[1])
        for i in range(5):
            if s1 > 0:
                strats[0][i] = strats[0][i]/s1
            if s2 > 0:
                strats[1][i] = strats[1][i]/s2

        return strats[0], strats[1]

    # Get the average strategy using the sum of every strategy
    def get_avg_strategy(self):
        avg_strategy = []
        for x in range(self.NUM_ACTIONS):
            avg_strategy.append(0)
        
        normalising_sum = 0
        for x in range(self.NUM_ACTIONS):
            normalising_sum += self.strategy_sum[x]

        for x in range(self.NUM_ACTIONS):
            if (normalising_sum > 0):
                avg_strategy[x] = self.strategy_sum[x] / normalising_sum
            else:
                avg_strategy[x] = 1.0 / self.NUM_ACTIONS

        return avg_strategy
        #return (self.print_avg_strategy(avg_strategy))
    
    # Creates a graph to show how the avg strategy changes with iterations
    def show_graph(self):
        plt.title("opponent strategy 0.2 0.2 0.2 0.2 0.2")
        plt.axis([None, None, -0.1, 1.1])
        plt.plot(self.rockstrats, label="rock")
        plt.plot(self.paperstrats, label="paper")
        plt.plot(self.scissorsstrats, label="scissors")
        plt.plot(self.lizardstrats, label="lizard")
        plt.plot(self.spockstrats, label="spock")
        plt.legend(loc='best')
        plt.show()
        pass
    
    # Prints out avg strategy in a readable way
    def print_avg_strategy(self):
        avg_strategy = self.get_avg_strategy()
        round_value = 5

        string = "Trainer Strategy \n"
        string += "Rock: "
        string += str(round(avg_strategy[0],round_value))
        # string += "\n"
        string += " Paper: "
        string += str(round(avg_strategy[1],round_value))
        # string += "\n"
        string += " Scissors: "
        string += str(round(avg_strategy[2],round_value))
        string += " Lizard: "
        string += str(round(avg_strategy[3],round_value))
        string += " Spock: "
        string += str(round(avg_strategy[4],round_value)) + "\n"

        return string
    
    # Prints out opponent strategy in readable way
    def print_opp_strategy(self):
        opps = self.opp_strategy
        string = "\nOpponent Strategy \n"
        string += "Rock: "
        string += str(opps[0]) + " "
        string += "Paper: "
        string += str(opps[1]) + " "
        string += "Scissors: "
        string += str(opps[2]) + " "
        string += "Lizard: "
        string += str(opps[3]) + " "
        string += "Spock: "
        string += str(opps[4]) + "\n"

        return string
    
# ----------------------------------------------------------------------------------- #
    
# Main method to run trainer
# use train() method to train player against a static opponent strategy
# use rps_to_nash() to train both player and opponent to obtain optimal RPS strategy
# use show_graph() when using train() only
# print_opp_strategy() and print_avg_strategy() to print out strategies

def main_method():
    # Input opponent strategy
    # opp_strategy = [0.4, 0.2, 0.2, 0.1, 0.1]
    opp_strategy = [0.2, 0.2, 0.2, 0.2, 0.2]
    trainer = rpslspTrainer(opp_strategy)
    start = time.time()
    #trainer.train(1000000)

    m = trainer.rps_to_nash(1000000)
    print("Player Strategy:", m[0])
    print("Opponent Strategy:", m[1])
    
    print(trainer.print_opp_strategy())
    print(trainer.print_avg_strategy())
    print("Time taken (s):", time.time() - start)
    trainer.show_graph()
    pass

main_method()