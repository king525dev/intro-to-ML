from numpy import *

def run():
     
     #Step 1 - Collect data
     points = getfromtxt('data.csv', delimeter=',')

     #Step 2 - Define our Hyperparameter
     learningRate = 0.0001 #How fast should our model converge
     #y = mx + c (Slope Formula)
     initialC = 0
     initialM = 0
     numOfIterations = 1000

     #Step 3 - Train our model
     print(f"Starting Gradient Descent at [ c ] = {initialC}, [ m ] = {initialM}, [ error ] = {computeErrorForLineGivenPoints(initialC, initialM, points)}")



if __name__ == '__main__':
     run();