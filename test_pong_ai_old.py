"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from wimblepong.agent import Agent
import torch
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("-m", "--model", help="Model file to load")
parser.add_argument("--cuda", default= True, action="store_true", help="Enable cuda")  #hua


args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu") #hua

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
#episodes = 100000
episodes = 1000000
TARGET_UPDATE_FREQ=1000  #hua
#LEARNING_STARTS = 59000
LEARNING_STARTS = 200

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
#player = wimblepong.SimpleAi(env, player_id)
player = Agent(env, env.observation_space.shape[0], player_id)

# Housekeeping
states = []
rewards = []
win1 = 0
step = 0
ballhitreward = 5
curPlayer = env.player1

if args.model:
    player.load_model(args.model)
for i in range(0,episodes):
    done = False
    player.reset()
    #player.epsilon = 200/(i+200)
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        #action1 = 0
        step += 1
        if step % 1000 == 0:
            print(step)
        if not args.model:    
            player.epsilon = max(0.1, 1 - step / 60000)
        #if step == 1 or step % 2 == 0:
        action1 = player.get_action()
        ob1, rew1, done, info = env.step(action1)
        if rew1 != 10 and rew1 != -10:
            #print(env.ball.y, env.player1.y)
            #sleep(1)
            #print(env.ball.x, curPlayer.x)
            #if curPlayer == env.player2:
            #    sleep(1)
            if env.ball.y < curPlayer.y + 14 and env.ball.y > curPlayer.y - 14 and abs(env.ball.x - curPlayer.x) < 10:
                rew1 = ballhitreward 
                #print(rew1)
        player.store_transition(player.state, action1, ob1, rew1, done)
        player.state = ob1
        rewards.append(rew1)
        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if step % 500 == 0:
            print("mean reward: ", np.mean(rewards[-500:]))    
        if not args.headless:
        #if step > 60000:
            env.render()
        if done:
            observation= env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            if i % 5 == 4:
                env.switch_sides()
                if curPlayer == env.player1:
                    curPlayer = env.player2
                else:
                    curPlayer = env.player1    
        if step < LEARNING_STARTS:
            continue
        if step % TARGET_UPDATE_FREQ == 0:
            player.replace_targetpolicy()            
        #if step % 4 == 0:
        if not args.model:
            player.update_network()
    if i % 50 == 0:
        player.save_model()            

