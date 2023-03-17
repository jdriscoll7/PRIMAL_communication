from tkinter import *
import random
import math
import time
from matplotlib.colors import hsv_to_rgb
import numpy as np
import os
import copy

DYNAMIC_TESTING=True
GOALS=True
output_path="environments"
model_path="model_primal"
dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
dir_english_dict = {0: "S", 1: "R", 2: "D", 3: "L", 4: "U"}

if DYNAMIC_TESTING:
    import tensorflow as tf
    from ACNet import ACNet
    
def init(data):
    data.communication_mode = False
    data.v_list = []
    data.a_dist = {}
    data.v = {}
    data.size=10
    data.state=np.zeros((data.size,data.size)).astype(int)
    data.goals=np.zeros((data.size,data.size)).astype(int)
    data.mode="obstacle"
    data.agent_counter=1
    data.primed_goal=0
    data.ID=0
    data.paused=True
    data.blocking_confidences=[]
    data.agent_goals=[]
    data.agent_positions=[]    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(output_path):
        for (_,_,files) in os.walk(output_path):
            for f in files:
                if ".npy" in f:
                    try:
                        ID=int(f[:f.find(".npy")])
                    except Exception:
                        continue
                    if ID>data.ID:
                        data.ID=ID
    data.ID+=1
    if DYNAMIC_TESTING:
        data.rnn_states=[]
        data.sess=tf.Session()
        data.network=ACNet("global",5,None,False,10,"global")
        #load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(data.sess,ckpt.model_checkpoint_path)        
        
def getDir(action):
    return dirDict[action]

def mousePressed(event, data):
    r=int((event.y/data.height)*data.state.shape[0])
    c=int((event.x/data.width)*data.state.shape[1])    
    if data.mode=="obstacle":
        if data.state[r,c]<=0 and data.goals[r,c]==0:
            data.state[r,c]=-((data.state[r,c]+1)%2)
    elif data.mode=="agent":
        if data.state[r,c]==0:
            data.state[r,c]=data.agent_counter
            data.goals[r,c]=data.agent_counter
            data.agent_positions.append((r,c))
            data.blocking_confidences.append(0)            
            data.rnn_states.append(data.network.state_init)
            data.agent_goals.append((r,c))
            data.agent_counter+=1
    elif data.mode=="goal":
        if data.state[r,c]>0 and data.primed_goal==0:
            data.primed_goal=data.state[r,c]
        elif data.state[r,c]!=-1 and data.primed_goal>0 and data.goals[r,c]==0:
            removeGoal(data,data.primed_goal)
            data.agent_goals[data.primed_goal-1]=(r,c)
            data.goals[r,c]=data.primed_goal
            data.primed_goal=0
            
def removeGoal(data,agent):
    for i in range(data.state.shape[0]):
        for j in range(data.state.shape[1]):
            if data.goals[i,j]==agent:
                data.goals[i,j]=0
                
def keyPressed(event, data):
    if event.keysym=='r':
        data.state=np.zeros((data.size,data.size)).astype(int)
        data.goals=np.zeros((data.size,data.size)).astype(int)
        data.agent_goals=[]
        data.rnn_states=[]
        data.agent_positions=[]        
        data.blocking_confidences=[]        
        data.primed_goal=0
        data.agent_counter=1
    elif event.keysym=="c":
        data.agent_counter=1
        data.primed_goal=0
        data.rnn_states=[]
        data.blocking_confidences=[]        
        data.agent_goals=[]
        data.agent_positions=[]
        data.goals=np.zeros((data.size,data.size))
        data.state=-(data.state==-1).astype(int)
    elif event.keysym=="p":
        data.paused=not data.paused
    elif event.keysym=="o":
        data.mode="obstacle"
    elif event.keysym=="g":
        data.mode="goal"
    elif event.keysym=="a":
        data.mode="agent"
    elif event.keysym=="q":
        data.communication_mode = not data.communication_mode
    elif event.keysym=='Up':
        data.size+=1
        data.state=np.zeros((data.size,data.size)).astype(int)
        data.goals=np.zeros((data.size,data.size)).astype(int)
    elif event.keysym=='Down':
        data.size-=1;
        if data.size<1:
            data.size==1
        data.state=np.zeros((data.size,data.size)).astype(int)
        data.goals=np.zeros((data.size,data.size)).astype(int)        
    elif event.keysym=="s":
        savedata=np.array([data.state,data.goals,data.agent_counter-1])
        np.save(output_path+"/%d"%data.ID,savedata)
        data.ID+=1
        
def observe(data,agent_id,goals):
    assert(agent_id>0)
    top_left=(data.agent_positions[agent_id-1][0]-10//2,data.agent_positions[agent_id-1][1]-10//2)
    bottom_right=(top_left[0]+10,top_left[1]+10)        
    obs_shape=(10,10)
    goal_map             = np.zeros(obs_shape)
    poss_map             = np.zeros(obs_shape)
    obs_map              = np.zeros(obs_shape)
    goals_map            = np.zeros(obs_shape)
    visible_agents=[]    
    for i in range(top_left[0],top_left[0]+10):
        for j in range(top_left[1],top_left[1]+10):
            if i>=data.state.shape[0] or i<0 or j>=data.state.shape[1] or j<0:
                #out of bounds, just treat as an obstacle
                obs_map[i-top_left[0],j-top_left[1]]=1
                continue
            if data.state[i,j]==-1:
                #obstacles
                obs_map[i-top_left[0],j-top_left[1]]=1
            if data.state[i,j]==agent_id:
                #agent's position
#                     pos_map[i-top_left[0],j-top_left[1]]=1
                poss_map[i-top_left[0],j-top_left[1]]=1
            elif data.goals[i,j]==agent_id:
                #agent's goal
                goal_map[i-top_left[0],j-top_left[1]]=1
            if data.state[i,j]>0 and data.state[i,j]!=agent_id:
                #other agents' positions
                poss_map[i-top_left[0],j-top_left[1]]=1
                visible_agents.append(data.state[i,j])                
    dx=data.agent_goals[agent_id-1][0]-data.agent_positions[agent_id-1][0]
    dy=data.agent_goals[agent_id-1][1]-data.agent_positions[agent_id-1][1]
    mag=(dx**2+dy**2)**.5
    if mag!=0:
        dx=dx/mag
        dy=dy/mag
    if goals:
        distance=lambda x1,y1,x2,y2:((x2-x1)**2+(y2-y1)**2)**.5
        for agent in visible_agents:
            x,y=data.agent_goals[agent-1]
            if x<top_left[0] or x>=bottom_right[0] or y>=bottom_right[1] or y<top_left[1]:
                #out of observation
                min_node=(-1,-1)
                min_dist=1000
                for i in range(top_left[0],top_left[0]+10):
                    for j in range(top_left[1],top_left[1]+10):
                        d=distance(i,j,x,y)
                        if d<min_dist:
                            min_node=(i,j)
                            min_dist=d
                goals_map[min_node[0]-top_left[0],min_node[1]-top_left[1]]=1
            else:
                goals_map[x-top_left[0],y-top_left[1]]=1
        return  ([poss_map,goal_map,goals_map,obs_map],[dx,dy,mag])
    else:
        return ([poss_map,goal_map,obs_map],[dx,dy,mag])


def agent_on_goal(data, agent_id):

    on_goal = False
    
    if data.goals[np.where(data.state == agent_id)] == agent_id:
        on_goal = True
        
    return on_goal
    

def number_to_base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def pad_list_zeros(input_list, n_agents):
    if len(input_list) < n_agents:
        return ([0]*(n_agents - len(input_list))) + input_list
    else:
        return input_list
        


def apply_action(data, agent, action):
    
    successful_action = False
    
    dx,dy = getDir(action)
    ax,ay = data.agent_positions[agent]
    if(ax+dx>=data.state.shape[0] or ax+dx<0 or ay+dy>=data.state.shape[1] or ay+dy<0):#out of bounds
        return successful_action
    if(data.state[ax+dx,ay+dy]<0):#collide with static obstacle
        return successful_action
    if(data.state[ax+dx,ay+dy]>0 and data.state[ax+dx,ay+dy] != agent + 1):
        return successful_action
    
    # No collision: we can carry out the action
    successful_action = True
    
    data.state[ax,ay] = 0
    data.state[ax+dx,ay+dy] = agent + 1
    data.agent_positions[agent] = (ax+dx,ay+dy)
    
    return successful_action


def apply_joint_action(data, actions):
    
    n_agents = len(data.agent_positions)
    
    # Table of action offsets for all agents.
    dx_list = [0 for _ in range(n_agents)]
    dy_list = [0 for _ in range(n_agents)]
    
    ax_list = [0 for _ in range(n_agents)]
    ay_list = [0 for _ in range(n_agents)]
    
    px_list = [0 for _ in range(n_agents)]
    py_list = [0 for _ in range(n_agents)]
    
    for ID, action in enumerate(actions):
        
        dx_list[ID], dy_list[ID] = getDir(action)
        ax_list[ID], ay_list[ID] = list(zip(*np.where(data.state == ID + 1)))[0]
        
        px_list[ID] = ax_list[ID] + dx_list[ID]
        py_list[ID] = ay_list[ID] + dy_list[ID]
        
        
    # Detect collisions, early return if detected.
    for ID, action in enumerate(actions):
        
        # Detect duplicate positions - doesn't count pass-throughs.
        p_list = list(zip(px_list, py_list))
        print(p_list)
        print("\n")
        print(list(zip(ax_list, ay_list)))
        print(list(zip(dx_list, dy_list)))
        if len(set(p_list)) !=  len(p_list):
            print("Collision detected - no move made. \n")
            return
    
    # Apply actions if no collisions detected.
    for ID, dx in enumerate(dx_list):
    
        ax = ax_list[ID]
        ay = ay_list[ID]
        dy = dy_list[ID]
    
        data.state[ax,ay] = 0
        data.state[ax+dx,ay+dy] = ID + 1
        data.agent_positions[ID] = (ax+dx,ay+dy)


def future_observe(data, agent_id, goals):
    
    n_agents = len(data.agent_positions)
    observations = [None for _ in range(5**n_agents)]

    # Keep dictionary for tracking temporary moves of agents.
    agent_moves = {move:0 for move in range(n_agents)}

    # Save current data state to restore later.
    save_state = copy.deepcopy(data.state)
    save_positions = copy.deepcopy(data.agent_positions)

    # Generate all 5^(n_agents) observations.
    for i in range(5**n_agents):
        
        valid_moves = [None for _ in range(n_agents)]
        
        #print(number_to_base(i, 5))
        
        for j, mod_state in enumerate(pad_list_zeros(number_to_base(i, 5), n_agents)):
            agent_moves[j] = mod_state
            
        # Apply the actions, make observation, undo the action.
        for agent, move_id in agent_moves.items():
            valid_moves[agent] = apply_action(data, agent, move_id)
        
        if False not in valid_moves:
            observations[i] = observe(data, agent_id, goals)
        
        data.state = copy.deepcopy(save_state)
        data.agent_positions = copy.deepcopy(save_positions)
    
    return observations
    

def timerFired(data):
    
    n_agents = len(data.agent_positions)
    actions = [0 for _ in range(n_agents)]
    actions_size = 5**n_agents
    
    data.v_list = [[-100 for _ in range(n_agents)] for _ in range(actions_size)]
    a_dist_list = [None for _ in range(actions_size)]
    rnn_state_list = [[None for _ in range(actions_size)] for _ in range(n_agents)]
    
    if DYNAMIC_TESTING and data.paused:
        for (x,y) in data.agent_positions:
            ID=data.state[x,y]
            observation=observe(data,ID,GOALS)
            rnn_state=data.rnn_states[ID-1]#yes minus 1 is correct
            a_dist,v,rnn_state,blocking = data.sess.run([data.network.policy,data.network.value,data.network.state_out,data.network.blocking], 
                                                   feed_dict={data.network.inputs:[observation[0]],
                                                            data.network.goal_pos:[observation[1]],
                                                            data.network.state_in[0]:rnn_state[0],
                                                            data.network.state_in[1]:rnn_state[1]})

            #data.blocking_confidences[ID-1]=np.ravel(blocking)[0]
            data.rnn_states[ID-1]=rnn_state
            
            #print(data.network.inputs)
            #print(observation[0])
            for i, f_observation in enumerate(future_observe(data,ID,GOALS)):
                if f_observation is not None:
                    a_dist_list[i], v_list_add, _, __ = data.sess.run([data.network.policy,data.network.value,data.network.state_out,data.network.blocking], 
                                                   feed_dict={data.network.inputs:[f_observation[0]],
                                                            data.network.goal_pos:[f_observation[1]],
                                                            data.network.state_in[0]:rnn_state[0],
                                                            data.network.state_in[1]:rnn_state[1]})
            
                    data.v_list[i][ID - 1] = v_list_add[0, 0]
                    
                    if (0 == pad_list_zeros(number_to_base(i, 5), n_agents)[ID-1]):
                        if not agent_on_goal(data, ID):
                            data.v_list[i][ID - 1] += -0.5
                    else:
                        data.v_list[i][ID - 1] += -0.3
                        
            
            data.a_dist[ID-1] = a_dist
            data.v[ID-1] = v
        
        print("\n----v_list----\n")
            
        for i, v_dist in enumerate(data.v_list):
            english_move_list = [dir_english_dict[k] for k in pad_list_zeros(number_to_base(i, 5), n_agents)]
            print(str(english_move_list) + ": " + str(v_dist) + "\t\t " + str(sum(v_dist)))
    
    if DYNAMIC_TESTING and not data.paused:
        
        for (x,y) in data.agent_positions:
            ID=data.state[x,y]
            observation=observe(data,ID,GOALS)
            rnn_state_old =data.rnn_states[ID-1]#yes minus 1 is correct
            a_dist,v,rnn_state,blocking = data.sess.run([data.network.policy,data.network.value,data.network.state_out,data.network.blocking], 
                                                   feed_dict={data.network.inputs:[observation[0]],
                                                            data.network.goal_pos:[observation[1]],
                                                            data.network.state_in[0]:rnn_state_old[0],
                                                            data.network.state_in[1]:rnn_state_old[1]})

            for i, f_observation in enumerate(future_observe(data,ID,GOALS)):
                    if f_observation is not None:
                        a_dist_list[i], v_list_add, rnn_state_list[ID-1][i], __ = data.sess.run([data.network.policy,data.network.value,data.network.state_out,data.network.blocking], 
                                                   feed_dict={data.network.inputs:[f_observation[0]],
                                                            data.network.goal_pos:[f_observation[1]],
                                                            data.network.state_in[0]:rnn_state_old[0],
                                                            data.network.state_in[1]:rnn_state_old[1]})
            
                        data.v_list[i][ID - 1] = v_list_add[0, 0]

                        if (0 == pad_list_zeros(number_to_base(i, 5), n_agents)[ID-1]):
                            if not agent_on_goal(data, ID):
                                data.v_list[i][ID - 1] += -0.5
                        else:
                            data.v_list[i][ID - 1] += -0.3

            
            
            data.rnn_states[ID-1]=rnn_state
            data.a_dist[ID-1] = a_dist
            data.v[ID-1] = v
            #data.blocking_confidences[ID-1]=np.ravel(blocking)[0]
            
            
        if n_agents > 0:
           
            if data.communication_mode:
                optimal_joint_action = np.argmax([sum(pair) for pair in data.v_list])
                actions = pad_list_zeros(number_to_base(optimal_joint_action, 5), n_agents)
                
                # Update rnn states for next computation.
                #for i, g in enumerate(actions):
                #    data.rnn_states[i] = rnn_state_list[i][optimal_joint_action]

            else:
                for agent_id in range(n_agents):
                    actions[agent_id] = np.argmax(data.a_dist[agent_id])
            
            
        print("\n----a_dist----\n")
        for i, a_dist in enumerate(a_dist_list):
            english_move_list = [dir_english_dict[k] for k in pad_list_zeros(number_to_base(i, 5), n_agents)]
            print(str(english_move_list) + ": " + str(a_dist) + "\t\t " + str(sum(data.v_list[i])))
            
            print("\n")
        
        print("\n\n----v_list----\n")
        
        for i, v_dist in enumerate(data.v_list):
            english_move_list = [dir_english_dict[k] for k in pad_list_zeros(number_to_base(i, 5), n_agents)]
            print(str(english_move_list) + ": " + str(v_dist) + "\t\t " + str(sum(v_dist)))
            
        
        if n_agents > 0:
            print("actions: " + str(actions))
            apply_joint_action(data, actions)


def redrawAll(canvas, data):
    #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    np.set_printoptions(precision=5)
    for r in range(data.state.shape[0]):
        y=(data.height/data.state.shape[0])*r
        color_depth=30
        for c in range(data.state.shape[1]):
            x=(data.height/data.state.shape[0])*c
            if data.state[r,c]==-1:
                canvas.create_rectangle(x, y, x+data.width/data.state.shape[0], y+data.height/data.state.shape[1],
                                            fill='grey', width=0)
            elif data.state[r,c]>0:
                color=hsv_to_rgb(np.array([(data.state[r,c]%color_depth)/float(color_depth),1,1]))
                color*=255
                color=color.astype(int)
                mycolor = '#%02x%02x%02x' % (color[0], color[1], color[2])
                canvas.create_rectangle(x, y, x+data.width/data.state.shape[0], y+data.height/data.state.shape[1],
                                                    fill=mycolor, width=0)  
                confidence=data.blocking_confidences[data.state[r,c]-1]
                confidence="%.0001f"%confidence
                #print("r: %d, c: %d, state: %f" % (r, c, data.state[r,c]))
                #a_dist = data.a_dist.setdefault(data.state[r,c]-1, np.zeros((1, 5))).T
                v = data.v.setdefault(data.state[r,c]-1, np.zeros((1, 5))).T
                #print("agent:" + str(data.state[r,c]) + ", action:" + str(a_dist.T) + "\n")
                canvas.create_text(x, y,
                                                    #fill='black', anchor="nw",text=a_dist,font="Arial 10 bold")
                                                    fill='black', anchor="nw",text=v,font="Arial 10 bold")                 
            if data.goals[r,c]>0:
                color=hsv_to_rgb(np.array([(data.goals[r,c]%color_depth)/float(color_depth),1,1]))
                color*=255
                color=color.astype(int)
                mycolor = '#%02x%02x%02x' % (color[0], color[1], color[2])
                if data.state[r,c]==data.goals[r,c]:
                    canvas.create_text(x+data.width/data.state.shape[0]/2, y+data.height/data.state.shape[1]/2,
                                                        fill="black", anchor="center",text="+",font="Arial 5 bold")
                else:
                    canvas.create_text(x+data.width/data.state.shape[0]/2, y+data.height/data.state.shape[1]/2,
                                                       fill=mycolor, anchor="center",text="+",font="Arial 5 bold")      
    for r in range(data.state.shape[0]):
        y=(data.height/data.state.shape[0])*r
        canvas.create_line(0,y,data.width,y,fill="black")
    for c in range(data.state.shape[1]):
        x=(data.height/data.state.shape[0])*c
        canvas.create_line(x,0,x,data.height,fill="black")
    canvas.create_text(data.width/2, 20,
                                fill="black", text=data.mode,font="Arial 20",anchor="center")
    txt="Paused" if data.paused else "Running"
    canvas.create_text(data.width-100, 20,
                       fill="black", text=txt,font="Arial 20",anchor="center")    
def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 200 # milliseconds
    init(data)
    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

def main():
    run(900,900)
if __name__=='__main__':
    main()
