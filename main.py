# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pygame
import random
import time
import tkinter as tk
import gym
import matplotlib.pyplot as plt 
import pandas as pd
#start UI
start_x=1
start_y=1
destination_x=23
destination_y=22
def show_entry_fields():
    global start_x
    start_x=int(e1.get())
    global start_y
    start_y=int(e2.get())
    global destination_x
    destination_x=int(e3.get())
    global destination_y
    destination_y=int(e4.get())

root = tk.Tk()
tk.Label(root, 
         text="Agent start x").grid(row=0)
tk.Label(root, 
         text="Agent start y").grid(row=1)
tk.Label(root, 
         text="Destination x").grid(row=2)
tk.Label(root, 
         text="Destination y").grid(row=3)

e1 = tk.Entry(root)
e2 = tk.Entry(root)
e3 = tk.Entry(root)
e4 = tk.Entry(root)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)

buttonOk = tk.Button(root, text="Ok", command=show_entry_fields).grid(row=5, column=0)
buttonQuit = tk.Button(root, text="Quit", command=root.destroy).grid(row=6, column=0)

root.mainloop()


#matris
matris = 0*np.ones((25, 25), dtype=int)
for i in range(0,300):
    random_x = random.randint(0, 24)
    random_y= random.randint(0, 24)
    matris[random_x][random_y]=-5
for i in range(0,25):
    matris[0][i]=-5
    matris[i][0]=-5
    matris[i][len(matris)-1]=-5
    matris[len(matris)-1][i]=-5
matris[start_x][start_y]=0
matris[destination_x][destination_y]=5
print(matris)



#matris txt
file = open("engel.txt", "w")
for i in range(0,25):
    file.write("\n")
    for j in range(0,25):
        if matris[i][j]==0:
            if i==start_x and j==start_y:
                file.write("({},{},A)".format(i+1,j+1))
            else:
                file.write("({},{},B)".format(i+1,j+1))
        elif matris[i][j]==-5:
            file.write("({},{},K)".format(i+1,j+1))
        elif matris[i][j]==5:
            file.write("({},{},D)".format(i+1,j+1))    
print(matris)        
file.close() 


start_rect_x=start_x
start_rect_y=start_y

liste_x=[]
liste_y=[]
RED=(255,0,0)
GREEN=(0,255,0)
BLUE=(0,0,255)
WHITE=(200,200,200)
YELLOW=(220,224,8)
BLACK=(0,0,0)

WIDTH =750
HEIGHT=750
FPS=30
pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("yazlab")
clock=pygame.time.Clock()

running=True
while running:
    clock.tick(FPS)   
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
    screen.fill(WHITE)
    
    
    for i in range (0,25):
        for j in range(0,25):
            if matris[i][j] == -5:
                pygame.draw.rect(screen, RED, pygame.Rect(j*30, i*30, 30, 30))
    for i in range (0,len(liste_x)):
        pygame.draw.rect(screen, BLUE, pygame.Rect(liste_y[i]*30+7, liste_x[i]*30+7, 15, 15))
    
    for i in range(0,750,30):
        pygame.draw.line(screen,BLACK,(i,0),(i,750))
        pygame.draw.line(screen,BLACK,(0,i),(750,i))
    pygame.draw.rect(screen, YELLOW, pygame.Rect(start_rect_y*30, start_rect_x*30, 30, 30))
    pygame.draw.rect(screen, GREEN, pygame.Rect(destination_y*30, destination_x*30, 30, 30))
    
    pygame.display.flip()

pygame.quit()
print("q learning basladi")








#Q learning



# Q table
q_table = np.zeros([len(matris)*len(matris),8])

# Hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.3

# Plotting Metrix
reward_list = []
all_costs = []
steps = []

episode_number = 75000
for i in range(1,episode_number):
    
    # initialize enviroment
    state_x =start_x
    state_y=start_y
    state=(state_x*len(matris))+state_y
    reward_count = 0
    dropouts = 0
    
    next_state_x=0
    next_state_y=0
    next_state=0
    reward=0
    done=False
    cost = 0
    xnxx = 0
    adim_maliyet=0.1
    while True:
        
        # exploit vs explore to find action
        # %30 = explore, %70 exploit
        if random.uniform(0,1) < epsilon:
            action = random.randint(0, 7)
        else:
            action = np.argmax(q_table[state])
        
        # action process and take reward/ observation
        if action==0:#up
            next_state_x=state_x-1
            next_state_y=state_y
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
        if action==1:#down
            next_state_x=state_x+1
            next_state_y=state_y
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
        if action==2:#right
            next_state_y=state_y+1
            next_state_x=state_x
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
        if action==3:#left
            next_state_y=state_y-1
            next_state_x=state_x
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
        if action==4:#ust sag çapraz
            next_state_y=state_y+1
            next_state_x=state_x-1
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
        if action==5:#sol ust çapraz
            next_state_y=state_y-1
            next_state_x=state_x-1
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
        if action==6:#sag alt çapraz
            next_state_y=state_y+1
            next_state_x=state_x+1
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
        if action==7:#sol alt çapraz
            next_state_y=state_y-1
            next_state_x=state_x+1
            if  next_state_x>(len(matris)-1) or next_state_x<0 or next_state_y>(len(matris)-1) or next_state_y<0:
                break;
            else:
                next_state=(next_state_x*len(matris))+next_state_y
                reward=matris[next_state_x][next_state_y]
                if matris[next_state_x][next_state_y]==-5 or matris[next_state_x][next_state_y]==5:
                    done=True
                
            
        
        
        # Q learning function
        old_value = q_table[state,action] # old_value
        
        xnxx += 1
        next_max = np.max(q_table[next_state]) # next_max
        
        next_value = (1-alpha)*old_value + alpha*(reward-adim_maliyet + gamma*next_max)
        
        # Q table update 
        q_table[state,action] = next_value
        cost+=reward
        # update state
        state = next_state
        state_x=next_state_x
        state_y=next_state_y
        # find wrong dropouts
        
            
        
        if done:
            all_costs.append(cost)
            steps += [xnxx]
            break
        reward_count+=reward
    if i%10==0:
        reward_list.append(reward_count)

#plot
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        #
ax1.plot(np.arange(len(steps)), steps, 'b')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Steps')
ax1.set_title('Episode via steps')

#
ax2.plot(np.arange(len(all_costs)), all_costs, 'r')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Cost')
ax2.set_title('Episode via cost')

plt.tight_layout()  # Function to make distance between figures

#
plt.figure()
plt.plot(np.arange(len(steps)), steps, 'b')
plt.title('Episode via steps')
plt.xlabel('Episode')
plt.ylabel('Steps')

#
plt.figure()
plt.plot(np.arange(len(all_costs)), all_costs, 'r')
plt.title('Episode via cost')
plt.xlabel('Episode')
plt.ylabel('Cost')

# Showing the plots
plt.show()


while True:
    konum=(start_x*len(matris))+start_y
    if np.argmax(q_table[konum])==0:#up
        start_x=start_x-1
        start_y=start_y
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))
    if np.argmax(q_table[konum])==1:#down
        start_x=start_x+1
        start_y=start_y
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))
    if np.argmax(q_table[konum])==2:#right
        start_y=start_y+1
        start_x=start_x
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))
    if np.argmax(q_table[konum])==3:#left
        start_y=start_y-1
        start_x=start_x
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))
    if np.argmax(q_table[konum])==4:#ust sag çapraz
        start_y=start_y+1
        start_x=start_x-1
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))
    if np.argmax(q_table[konum])==5:#sol ust çapraz
        start_y=start_y-1
        start_x=start_x-1
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))
    if np.argmax(q_table[konum])==6:#sag alt çapraz
        start_y=start_y+1
        start_x=start_x+1
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))
    if np.argmax(q_table[konum])==7:#sol alt çapraz
        start_y=start_y-1
        start_x=start_x+1
        liste_x.append(start_x)
        liste_y.append(start_y)
        #print("({},{})".format(start_x,start_y))

    if destination_x==start_x and destination_y==start_y:
        break;









#---UI map---
#colors
RED=(255,0,0)
GREEN=(0,255,0)
BLUE=(0,0,255)
WHITE=(200,200,200)
YELLOW=(220,224,8)
BLACK=(0,0,0)

WIDTH =750
HEIGHT=750
FPS=30
pygame.init()
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("yazlab")
clock=pygame.time.Clock()

running=True
while running:
    clock.tick(FPS)   
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
    screen.fill(WHITE)
    
    
    for i in range (0,25):
        for j in range(0,25):
            if matris[i][j] == -5:
                pygame.draw.rect(screen, RED, pygame.Rect(j*30, i*30, 30, 30))
    
    
    for i in range(0,750,30):
        pygame.draw.line(screen,BLACK,(i,0),(i,750))
        pygame.draw.line(screen,BLACK,(0,i),(750,i))
    pygame.draw.rect(screen, YELLOW, pygame.Rect(start_rect_y*30, start_rect_x*30, 30, 30))
    pygame.draw.rect(screen, GREEN, pygame.Rect(destination_y*30, destination_x*30, 30, 30))
    for i in range (0,len(liste_x)):
        pygame.draw.rect(screen, BLUE, pygame.Rect(liste_y[i]*30+7, liste_x[i]*30+7, 15, 15))
    pygame.display.flip()

pygame.quit()

