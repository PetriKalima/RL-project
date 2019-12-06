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
from wimblepong.agenthit import Agent
import torch
from time import sleep
#from utils import plot_rewards

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
episodes = 1000000 # org 0512
#episodes = 1000

TARGET_UPDATE_FREQ=1000  #hua
LEARNING_STARTS = 59000  #org 0512
#LEARNING_STARTS = 2000  # for testing fast

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
#player = wimblepong.SimpleAi(env, player_id)
player = Agent(env, env.observation_space.shape[0], player_id)

# Training loop
cumulative_rewards = []
#  for Priority Memory
statesP=[]  # for prio
actionsP=[]
ob1sP=[] 
rew1sP=[]
donesP=[]
flagwin_epoNotadded= True  # for prio

# Housekeeping
states = []
rewards = []
win1 = 0
step = 0
#ballhitreward = 5
ballhitreward = 1
curPlayer = env.player1


if args.model:
    player.load_model(args.model)
for i in range(0,episodes):
    done = False
    player.reset()
    ball_pre_x=env.ball.x  #add 0612
    cum_reward = 0
    #player.epsilon = 200/(i+200)

    state=player.state # for Prio

    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        #action1 = 0
        step += 1
        if step % 1000 == 0:
            print(step)
        if not args.model:    
            player.epsilon = max(0.1, 1 - step / 1000000) # 0512 org
            #player.epsilon = max(0.1, 1 - step / 5000)  # for short testing
        if step == 1 or step % 2 == 0:
            action1 = player.get_action()
        ob1, rew1, done, info = env.step(action1)
        cum_reward += rew1  #add 0512
        
        #for Priority memory start
        statesP.append(state)        
        actionsP.append(action1)
        ob1sP.append(ob1) 
        rew1sP.append(rew1)
        donesP.append(done)
        # for priority memory end

        if rew1 != 10 and rew1 != -10:
            #print(env.ball.y, env.player1.y)
            #sleep(1)
            #print(env.ball.x, curPlayer.x)
            #if curPlayer == env.player2:
            #    sleep(1)
            if env.ball.y < curPlayer.y + 14 and env.ball.y > curPlayer.y - 14 and abs(env.ball.x - curPlayer.x) < 10:
                if curPlayer.x <=10 and (env.ball.x>= ball_pre_x):  # add 0612
                   rew1 = ballhitreward 
                if curPlayer.x >=120 and (env.ball.x<= ball_pre_x):   
                   rew1 = ballhitreward
                #print(rew1)
        #player.store_transition(player.state, action1, ob1, rew1, done)  #  mod for Priority memory
        player.state = ob1
        state=ob1  # for priority memory
        ball_pre_x=env.ball.x #add 0612

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
        '''
        if step % 4 == 0:  # original  0512
        
            player.update_network()
        '''    
        player.update_network() 

    cumulative_rewards.append(cum_reward) 
    #plot_rewards(cumulative_rewards)      
    #if i % 4 == 0:
    #    player.update_network()            
        #player.target_net.load_state_dict(player.policy_net.state_dict())

    # for priority memory


    print("rew1, cum_reward:", rew1,cum_reward)
    
    
    if done and rew1 ==10 and flagwin_epoNotadded:  # need to check startegy
       for k in range(len(statesP)):
          player.store_transition(statesP[k], actionsP[k], ob1sP[k], rew1sP[k], donesP[k])
       flagwin_epoNotadded = False

    if done and rew1 ==-10 and (flagwin_epoNotadded==False ):  # need to check startegy
       for k in range(len(statesP)):
          player.store_transition(statesP[k], actionsP[k], ob1sP[k], rew1sP[k], donesP[k])
       flagwin_epoNotadded= True

    statesP.clear()        
    actionsP.clear() 
    ob1sP.clear() 
    rew1sP.clear() 
    donesP.clear()         
        



    # for Priority memory save end
      


    if i % 300:
        #plt.plot(cumulative_rewards)

        plt.figure(2)
        plt.clf()
        rewards_t = torch.tensor(cumulative_rewards, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative reward')
        plt.grid(True)
        plt.plot(rewards_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            #means=means.cpu()
            plt.plot(means.numpy())

        plt.savefig("plot.png")
        
    if i % 50 == 0:
        player.save_model()            

print('Complete')
plt.ioff()
plt.show()
