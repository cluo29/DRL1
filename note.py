#basic array and matrix control

import numpy as np



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