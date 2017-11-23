

#basic array and matrix control

import numpy as np


def executeAction(action,  L, F, T):
    L_Next = 0
    F_Next = 0
    T_Next = T
    if action in [0,3,6]:
        L_Next = L
    elif action in [1,4,7]:
        L_Next = L + 1
    elif action in [2,5,8]:
        L_Next = L - 1
    if L_Next == 3:
        L_Next = 2
    elif L_Next == 0:
        L_Next = 1

    if action in [0, 1, 2]:
        F_Next = F
    elif action in [3,4,5]:
        F_Next = F + 1
    elif action in [6,7,8]:
        F_Next = F -1
    if F_Next == 3:
        F_Next = 2
    elif F_Next == -1:
        F_Next = 0

    if L_Next == 2:
        T_Next = T_Next + 5
    else:
        T_Next = T_Next + 3
    if F_Next ==2:
        T_Next = T_Next - 4
    elif F_Next ==1:
        T_Next = T_Next - 2


    reward =0
    if T_Next<= 70:
        reward = L_Next + 2 - F_Next
    else:
        reward = 70 - T_Next + L_Next + 2 - F_Next

    return L_Next, F_Next, T_Next, reward

action = 1
if action in [0,1,2]:
    print("yes")

#[[0 0 0]]
stateNP = np.random.randint(0, 1, size=(1, 3))
#print(stateNP)

#random 0 to 2 , [[2 2 2]]
stateNP = np.random.randint(0, 3, size=(1, 3))
print(stateNP)

print("-----")

#get cell
print("stateNP 1st")
print(stateNP[0][0])
print("stateNP 2nd")
print(stateNP[0][1])
print("stateNP 3rd")
print(stateNP[0][2])

print("-----")
#from [[1 1 2]]
#to [[[1 1 1]
#  [1 1 1]
#  [2 2 2]]]
stateNP = np.stack([stateNP] * 3, axis=2)
print(stateNP)
print("-----")
# get column 3
stateVectorNow = stateNP[:, :, 2]
print(stateVectorNow)
print("-----")
#get cell
print("stateVectorNow 1st")
print(stateVectorNow[0][0])
print("stateVectorNow 2nd")
print(stateVectorNow[0][1])
print("stateVectorNow 3rd")
print(stateVectorNow[0][2])

L_Next, F_Next, T_Next,reward = executeAction(8, 2,1,150)
print("L_Next")
print(L_Next)
print("F_Next")
print(F_Next)
print("T_Next")
print(T_Next)
print("Reward")
print(reward)