from numpy import *

def computeErrorForLineGivenPoints(c, m, points):
     #Initialize Error at 0
     totalError = 0

     for i in range(0, len(points)):
          #Get 'x' value
          x = points[i, 0]
          #Get 'y' value
          y = points[i, 1]
          #Get differemce , square it, then add it to the total
          totalError += (y -(m * x + c)) ** 2

     #Return the Average
     return totalError / float(len(points))

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
     [c, m] = gradientDescentRunner(points, initialC, initialM, learningRate, numOfIterations)
     print(f"Ending Point at [ c ] = {c}, [ m ] = {m}, [ error ] = {computeErrorForLineGivenPoints(c, m, points)}")



if __name__ == '__main__':
     run();