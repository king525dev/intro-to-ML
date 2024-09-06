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
          totalError += (y - ((m * x) + c)) ** 2

     #Return the Average
     return totalError / float(len(points))

def gradientDescentRunner(points, startingC, startingM, learningRate, numOfIterations):
     #Starting c and m
     c = startingC
     m = startingM

     #Gradient Desc
     for i in range(numOfIterations):
          #Update c and m with the new, more accurate, c and m by performing
          #this gradient step
          c, m = stepGradient(c, m, array(points), learningRate)

     #Return Optimal value
     return [c, m]

#The Magic
def stepGradient(currentC, currentM, points, learningRate):

     #Starting Point for our gradients
     gradientC = 0
     gradientM = 0

     N = float(len(points))

     for i in range(0 ,len(points)):
          #Get 'x' value
          x = points[i, 0]
          #Get 'y' value
          y = points[i, 1]

          #Direction with respect to c and m
          #Computing Partial derivatives of out error function
          gradientC += -(2 / N) * (y - ((currentM * x) + currentC))
          gradientM += (2 / N) * x * (y - ((currentM * x) + currentC))

     #Update c and m values using our partial derivatives
     newC = currentC - (learningRate * gradientC)
     newM = currentM - (learningRate * gradientM)

     return [newC, newM]

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