import numpy as np
import math
import random
import matplotlib.pyplot as plt

trainInput  = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]
                ],dtype=float)
trainOutput = np.array([[0.0],[1.0],[1.0],[0.0]
                ],dtype=float)
class NN:
    INPUT_NEURONS   = 2
    HIDDEN_NEURONS  = 2
    OUTPUT_NEURONS  = 1
    MAX_SAMPLES     = 4

class HS:
    HM    = 10
    HMCR    = 0.7
    PAR     = 0.3
    UPPER   = 5.0
    LOWER   = -5.0
    B       = (UPPER-LOWER)/10*random.uniform(0,2)
    PRACTICE= 2000
    ERR_THRES   = 0.001
    
#initialize all the variable needed for ANN
wih = np.ndarray((NN.INPUT_NEURONS+1,NN.HIDDEN_NEURONS),float) #this is input for HS
who = np.ndarray((NN.HIDDEN_NEURONS+1,NN.OUTPUT_NEURONS),float)#this is input for HS

inputs      = np.ndarray((NN.INPUT_NEURONS),float)
hidden      = np.ndarray((NN.HIDDEN_NEURONS,NN.OUTPUT_NEURONS),float)
target      = np.ndarray((NN.OUTPUT_NEURONS),float)
actual      = np.ndarray((NN.OUTPUT_NEURONS),float)
#untuk masuk ke Harmony Memory
error       = np.ones((HS.HM,),float)

hs_input_size   = wih.size + who.size #total neuron + bias
harmony_memory  = np.ndarray((HS.HM,hs_input_size),float)

input_hidden = NN.INPUT_NEURONS * NN.HIDDEN_NEURONS
hidden_output= NN.HIDDEN_NEURONS * NN.OUTPUT_NEURONS
#=================================================
#Harmony Search Function

#init weight for the first time
def assign_random():
    vector      = np.ndarray((hs_input_size,),float)
    for i in range(hs_input_size):
        vector[i] = random.uniform(HS.LOWER,HS.UPPER)
    return vector

def vector_to_weight(vector):
    #split into array of weight, ex. [w11,w12] = [1,0.3]
    pointer = 0
    for i in range(NN.INPUT_NEURONS+1):
        for j in range(NN.HIDDEN_NEURONS):
            wih[i][j] = vector[pointer]
            pointer+=1
    for i in range(NN.HIDDEN_NEURONS+1):
        for j in range(NN.OUTPUT_NEURONS):
            who[i][j] = vector[pointer]
            pointer+=1

#fitness function
def mse():
    er = 0.0
    for i in range(NN.OUTPUT_NEURONS):
        er += pow((target[i]-actual[i]),2)
    error1 = 1/NN.OUTPUT_NEURONS*er
    return error1

#this is HARMONY!
def weight_adjust():
    xi = np.ndarray((hs_input_size,),float)
    for j in range(hs_input_size):
        r = random.random()
        if(r <= HS.HMCR):
            rand = random.randrange(0,HS.HM)
            xi[j]=(harmony_memory[rand][j])
            rand = random.random()
            if(rand <= HS.PAR):
                roperator = random.randrange(0,1)
                if(roperator == 1):
                    xi[j] = xi[j] + HS.B
                else: 
                    xi[j] = xi[j] - HS.B
                
                if(xi[j] > HS.UPPER):
                    xi[j] = HS.UPPER
                elif(xi[j] < HS.LOWER):
                    xi[j] = HS.LOWER
        else:
            xi[j]=(HS.LOWER + random.uniform(0,HS.UPPER-HS.LOWER))
    vector_to_weight(xi)
    resultxi = MLP_train()
    if(resultxi < max(error)):
        index, = np.where(error == max(error))
        harmony_memory[index[0]] = xi
        error[index[0]] = resultxi
    # print(min(error))
    print('min: '+ str(min(error)))
    best, = np.where(error == min(error))
    return (min(error),best)

def init_harmony_memory():
    for i in range(HS.HM):
        harmony_memory[i] = assign_random()
        vector_to_weight(harmony_memory[i])
        error[i] = MLP_train()

#End here ---------- Harmony Search Function
#=================================================
#MLP Function
def feed_fordward():
    #count sig(x) from input layer to hidden layer
    for hid in range(NN.HIDDEN_NEURONS):
        sum = 0
        for inp in range(NN.INPUT_NEURONS):
            sum += inputs[inp] * wih[inp][hid]
        sum += 1*wih[NN.INPUT_NEURONS][hid] #bias
        hidden[hid] = sigmoid(sum)
        # print("Hidden "+str(hid)+" "+str(inp)+": "+str(hidden[hid]))
    #count sig(x) from  hidden layer to output layer
    for out in range(NN.OUTPUT_NEURONS):
        sum = 0
        for hid in range(NN.HIDDEN_NEURONS):
            sum += hidden[hid][0] * who[hid][out]
        sum += who[NN.HIDDEN_NEURONS][out] #bias
        actual[out] = sigmoid(sum)
    return mse()

def predict():
    #count sig(x) from input layer to hidden layer
    for hid in range(NN.HIDDEN_NEURONS):
        sum = 0
        for inp in range(NN.INPUT_NEURONS):
            sum += inputs[inp] * wih[inp][hid]
        sum += wih[NN.INPUT_NEURONS][hid] #bias
        hidden[hid] = sigmoid(sum)

    #count sig(x) from  hidden layer to output layer
    for out in range(NN.OUTPUT_NEURONS):
        sum = 0
        for hid in range(NN.HIDDEN_NEURONS):
            sum += hidden[hid] * who[hid][out]
        sum += who[NN.HIDDEN_NEURONS][out] #bias
        actual[out] = sigmoid(sum)
    return actual


#activation function
def sigmoid(val):
    return 1.0/(1.0+math.exp(-val))

def MLP_train():
    sum_error = 0.0
    for i in range(NN.MAX_SAMPLES):
        #assign trainInput into train
        for j in range(NN.INPUT_NEURONS):
            inputs[j] = trainInput[i][j]
        #assign trainOutput into target
        for j in range(NN.OUTPUT_NEURONS):
            target[j] = trainOutput[i][j]
        sum_error += feed_fordward()
    return sum_error/NN.MAX_SAMPLES
    
#End here ---------- MLP Function
#=================================================


#define the plot first
plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.xlim(0,HS.PRACTICE)
plt.ylim(0,1)
plt.draw()

best_weight =[]
init_harmony_memory()
for i in range(HS.PRACTICE):
    best,idx = weight_adjust()
    best_weight = harmony_memory[idx]

    #show in scatter
    x.append(i)
    y.append(best)
    sc.set_offsets(np.c_[x,y])
    fig.canvas.draw_idle()
    plt.pause(0.01)

vector_to_weight(best_weight[0])
print('wih')
print(wih)
print('-----------')
print('who')
print(who)
inputs =[0.0,0.0]
print(predict())