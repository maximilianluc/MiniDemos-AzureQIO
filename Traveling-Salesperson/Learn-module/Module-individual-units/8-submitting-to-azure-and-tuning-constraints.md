The final task to complete the navigation system is to tune the constraint weights. To do this, you will need to submit the problem to Azure Quantum workspace, and evaluate the results returned by the solver. In the previous unit, verification steps were covered to check the validity of a solution configuration. For tuning, these steps will save a lot of investigatory work. Because tuning is an iterative process and dependent on many factors (solvers, constraints, cost definitions, etc.) we will only implement the **simulated annealing** solver, but feel free to uncomment other solver methods in the code and experiment!


## The final cost function

The optimization problem, which you have programmed into the navigation system in the previous units, will be submitted to the Azure Quantum solvers to find efficient routes through the solar system! Before that, you will need to assign a value to the weights $w_1$, $w_2$, $w_3$, $w_4$, $w_5$. 

$$ \text{Cost function so far: } \underset{x_0, x_1,\dots,x_{(N^2+2N)}}{min} \left(\sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right)  +   w_1 \left( \sum_{l=0}^{N} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} x_{(i+Nl)} \cdot x_{(j+Nl)} \text{ with } \{ i,j | i<j \} \right) +w_2\left( \sum_{k=0}^{N(N+1)-1} x_k \right) + w_3 \left(  \large{\sum}_{p=0}^{N^2+N-1}\hspace{0.25cm} \large{\sum}_{f=p+N,\hspace{0.2cm} \text{stepsize: } N}^{N^2-1} \hspace{0.35cm} (x_p \cdot x_f)   \right)  + w_4 ( x_0)  + w_5 ( x_{N^2}) \right)  $$

Remember that $w_1$ and $w_3$ should be a positive value (a cost) and that $w_2$, $w_4$, and $w_5$ should be a negative value (negative cost, reward).

### Initial weighting

The travel costs of the spaceship can change when the space rocks need to be mined or sold on different planets. For the cost function it is important to take this into account, as you want the travel costs and constraints to have a constant relative weighting. Otherwise it could be the case that the solver punishes a constraint much less strictly, resulting in poor routes. Weighting the constraints as a function of the cost matrix as well, like the travel costs, will allow you to have constant relative weighting in the cost function. Then regardless of the distances between planets, moons, and asteroids considered, the navigation system should generate efficient routes. 

There are a number of ways to implement a relative weighting, which boils down to a design decision. Here, the constraint weights are tuned by multiplying some factor $\alpha$ by the maximum element in the cost matrix $C$:

$w_h = \alpha \cdot \max(C_{i,j}).$

Other functions, for example matrix norms, are good candidates as well. You will need to re-tune the constraints if you decide to change them!


- Some tips for the tuning process:
  - Start by assigning the start and end locations $w_4$ and $w_5$ with a big negative weight (but keep $\alpha \leq$  10). Once this constraint is always satisfied, work on the other constraint weights.
  - Look at the printed route, and see which constraint it violates. Then try to tune the respective constraint(s) accordingly. Iteratively try to do this and you will eventually find good routes!


## Submitting the code

In the code below, the constraints weights have all been initialized dependent on the maximum element in the cost matrix. It is your job to experiment with around with the weights given in the constraint ($w_1$, $w_2$, $w_3$, $w_4$, $w_5$). If you want to skip the tuning process, you can use the values below:

- $w_1: int(2 \cdot np.max(CostMatrix))$
- $w_2: int(-1.65 \cdot np.max(CostMatrix))$
- $w_3: int(2 \cdot np.max(CostMatrix))$
- $w_4: int(-10 \cdot np.max(CostMatrix))$
- $w_5: int(-10 \cdot np.max(CostMatrix))$

The code is submitted to the Azure solvers as a `synchronous` job because you will be manually tuning and evaluating the results sequentially. Another option is submit the jobs `asynchronously`, which is helpful if you want to solve many different problem instances and compare them afterwards. Asynchronous submissions are especially helpful if you want to automate parameter tuning through grid searches, for example.

>[!Note]
>It is good practice to use `int`s instead of `float`s for the cost function weights. Doing so will give more accurate results. It is not mandatory, however.

``` python

### FILL IN THE CONSTRAINT WEIGHTS w_1, w_2, w_3, w_4, AND w_5 IN THE PROBLEM DEFINITION TO SUBMIT!

from azure.quantum import Workspace
from azure.quantum.optimization import Problem, ProblemType, Term, HardwarePlatform, Solver
from azure.quantum.optimization import SimulatedAnnealing, ParallelTempering, Tabu, QuantumMonteCarlo
from typing import List

import numpy as np
import math

workspace = Workspace (
    subscription_id = "",  # Add your subscription id
    resource_group = "",   # Add your resource group
    name = "",             # Add your workspace name
    location = ""          # Add your workspace location (for example, "westus")
)

workspace.login()

##### Define variables

# The number of planets/moons/asteroids.
NumLocations = 10

# Location names. Names of some of the solar system's planets/moons/asteroids.
LocationNames = {0:'Mars', 1:'Earth', 2:"Earth's Moon", 3:'Venus', 4:'Mercury', 5:'Ganymede', 6:'Titan', 7:'Ceres', 8:'Pallas', 9:'Cybele'}

# Approximate mean distances between the planets/moons/asteroids. Note that they can be very innacurate as orbital mechanics are ignored. 
# This is a symmetric matrix since we assume distance between planets is constant for this module.
CostMatrix = np.array([     [0,   78,       2,  120,   170,  550,  1200,  184, 600,  1.5   ],
                            [78,   0,     0.5,   41,    92,  640,  1222,  264, 690,  0.25  ],
                            [2,   0.5,      0,   40,    91,  639,  1221,  263, 689,  0.25  ], 
                            [120,  41,     40,    0,    50,  670,  1320,  300, 730,  41.5  ],
                            [170,  92,     91,   50,     0,  720,  1420,  400, 830,  141.5 ],
                            [550,  640,   639,  670,   720,    0,   650,  363,  50,  548   ],
                            [1200, 1222, 1221, 1320,  1420,  650,     0, 1014,  25,  625   ],  
                            [184,  264,   263,  300,   400,  363,  1014,    0, 100,  400   ],
                            [600,  690,   689,  730,   830,   50,    25, 100,    0,  350   ],
                            [1.5,  0.25, 0.25, 41.5, 141.5,  548,   625, 400,  350,  0     ]
                      ])    
                       
##### If you want try running with a random cost matrix, uncomment the following:
#maxCost = 10
#CostMatrix = np.random.randint(maxCost, size=(NumLocations,NumLocations))
 
############################################################################################
##### Define the optimization problem for the Quantum Inspired Solver
def OptProblem(CostMatrix) -> Problem:
    
    #'terms' will contain the weighting terms for the solver!
    terms = []

    ############################################################################################
    ##### Cost of traveling between locations  
    for k in range(0,len(CostMatrix)):                          # For each trip (there are N trips to pass through all the locations and return to home base)
        for i in range(0,len(CostMatrix)):                      # For each origin (reference location)
            for j in range(0,len(CostMatrix)):                  # For each destination (next location w.r.t reference location)
                
                #Assign a weight to every possible trip from location i to location j - for any combination
                terms.append(
                    Term(
                        c = CostMatrix.item((i,j)),                                     # Element of the cost matrix
                        indices = [i+(len(CostMatrix)*k), j+(len(CostMatrix)*(k+1))]    # +1 to denote dependence on next location
                    )
                )
                ##----- Uncomment one of the below statements if you want to see how the weights are assigned! -------------------------------------------------------------------------------------------------
                #print(f'{i+(len(CostMatrix)*k)}, {j+(len(CostMatrix)*(k+1))}')                                                                 # Combinations between the origin and destination locations 
                #print(f'For x_{i+(len(CostMatrix)*k)}, to x_{j+(len(CostMatrix)*(k+1))} in trip number {k} costs: {CostMatrix.item((i,j))}')   # In a format for the solver (as formulated in the cost function)
                #print(f'For location_{i}, to location_{j} in trip number {k} costs: {CostMatrix.item((i,j))}')                                         # In a format that is easier to read for a human

    ############################################################################################
    ##### Constraint: One location at a time constraint - spaceship can only be in 1 location at a time.
    for l in range(0,len(CostMatrix)+1):                # The total number of locations that are visited over the route (N+1 because returning to home base)
        for i in range(0,len(CostMatrix)):              # For each location (iterate over the location vector)
            for j in range(0,len(CostMatrix)):          # For each location (iterate over the location vector)
                if i!=j and i<j:                        # i<j because we don't want to penalize twice // i==j is forbidden (this could equal 1, that's why)
                    terms.append(
                        Term(
                            c = int(w_1 * np.max(CostMatrix)),       # FILL IN - w_1 should be a positive value
                            indices = [i+(len(CostMatrix)*l),j+(len(CostMatrix)*l)]                   
                        )
                    )
                    ##----- Uncomment one of the below statements if you want to see how the weights are assigned! -------------------------------------------------------------------------------------------------
                    #print(f'{i+(len(CostMatrix)*l)},{j+(len(CostMatrix)*(l))}')
                    #print(f'Location constraint 1: x_{i+(len(CostMatrix)*l)} - x_{j+(len(CostMatrix)*(l+1))} (trip {l}) assigned weight: FILL IN')  # In a format for the solver (as formulated in the cost function)

    ############################################################################################
    ##### Constraint: No dissapearing constraint - encourage the spaceship to be 'somewhere' otherwise all x_k might be 0 (for example).
    for v in range(0, len(CostMatrix) + len(CostMatrix) * (len(CostMatrix))):    # Select variable 
        terms.append(
            Term(
                c =  int(w_2 * np.max(CostMatrix)),          # FILL IN - w_2 should be a negative value
                indices = [v]   
            )
        )
        ##----- Uncomment one of the below statements if you want to see how the weights are assigned! -------------------------------------------------------------------------------------------------
        #print(v)
        #print(f'No dissapearing constraint 2: x_{v} assigned weight: FILL IN')                                                      # In a format for the solver (as formulated in the cost function)
        #print(f'No dissapearing constraint 2: location_{v % NumLocations} after {np.floor(v / NumLocations)} trips assigned weight: FILL IN')   # In a format that is easier to read for a human

    ############################################################################################                        
    ##### Constraint: no revisiting locations constraint --- (in the last step we can travel without penalties (this is to make it easier to specify an end location ))
    for p in range(0,len(CostMatrix)+len(CostMatrix)*(len(CostMatrix))):                                  # This selects a present location x: 'p' for present    
        for f in range(p+len(CostMatrix),len(CostMatrix)*(len(CostMatrix)),len(CostMatrix)):              # This selects the same location x but after upcoming trips: 'f' for future
            terms.append(
                Term(
                    c = int(w_3 * np.max(CostMatrix)),  # FILL IN - w_3 should be a positive value
                    indices = [p,f]   
                )
            )     
            ##----- Uncomment one of the below statements if you want to see how the weights are assigned! -------------------------------------------------------------------------------------------------
            #print(f'x_{p},x_{f}')  # Just variable numbers 
            #print(f'Visit once constraint: x_{p} - x_{f}  assigned weight: FILL IN')  # In a format for the solver (as formulated in the cost function)
            #print(f'Visit once constraint: location_{p%NumLocations} - location_{(p+f)%NumLocations} after {(f-p)/NumLocations} trips assigned weight: FILL IN')  # In a format that is easier to read for a human
            
    #############################################################################################                        
    ##### Begin at x_0 (Mars - home base)
    terms.append(
        Term(
            c = int(w_4 * np.max(CostMatrix)),  # FILL IN - w_4 should be a negative value,
            indices = [0]   
        )
    )

    ############################################################################################                        
    ##### End at x_{N^2} (Mars - home base)
    terms.append(
        Term(
            c = int(w_5 * np.max(CostMatrix)),  # FILL IN - w_5 should be a negative value,
            indices = [len(CostMatrix)*(len(CostMatrix))]   
        )    
    )

    return Problem(name="Spaceship navigation system", problem_type=ProblemType.pubo, terms=terms)
    
def ReadResults(Config: dict, LocationNames, CostMatrix, NumLocations):  

    #############################################################################################
    ##### Read the return result (dictionary) from the solver and sort it
    RouteChoice = Config.items()
    RouteChoice = [(int(k), v) for k, v in Config.items()] 
    RouteChoice.sort(key=lambda tup: tup[0]) 

    #############################################################################################
    ##### Initialize variables to understand the routing    
    TimeStep=[]                                      # This will contain an array of times/trips - each location is represented for each time/trip interval
    Names = []                                       # This will contain an array of location names (string)
    Location = []                                    # This will contain the location(s) the spaceship for each time/trip (numerical, 0 or 1)
    RouteMatrixElements = []                         # This will contain the indices of the cost matrix representing where the spaceship has traveled (to determine total cost)

    #############################################################################################
    ##### Go through locations during each timestep/trip (all x_v) to see where the spaceship has been
    for Index in RouteChoice:
        TimeStep.append(math.floor(Index[0] / len(CostMatrix)))         # Time step/trip = the k-th is floor of the index divided by the number of locations
        Names.append(LocationNames[(Index[0] % len(CostMatrix))])       # Append location names for each time step
        Location.append(Index[1])                                       # Append location for each time step (0 or 1 if spaceship has been there or not)
        if Index[1] == 1:                                               # Save selected location where the spaceship travels to in that trip (given by x_v = 1)
            RouteMatrixElements.append(Index[0] % len(CostMatrix))      # Save the indices (this returns the row index)
    SimulationResult = np.array([TimeStep, Names, Location])            # Save all the route data (also where the spaceship did not go during a turn/trip/timestep)
 
    #############################################################################################
    ##### Create the route dictionary 
    k=0                                                                                                             
    RouteDict = {}                                                                                                                                              
    RouteDict['Route'] = {}
    RouteMatrix = np.array([['Timestep,', 'Location']])
    for i in range(0, (NumLocations * (NumLocations + 1))):
        if SimulationResult[2][i] == '1':                                                                                     # If the SimulationResult[2][i] (location) == 1, then that's where the spaceship goes/went
            RouteMatrix = np.concatenate((RouteMatrix, np.array([[SimulationResult[j][i] for j in range(0, 2)]])), axis=0)    # append the rows where the spaceship DOES travel to to RouteMatrix
            RouteDict['Route'].update({k: RouteMatrix[k + 1][1]})                                                             # Save the route to a dictionary
            k += 1                                                                                                            # Iterable keeps track for the dictionary, but also allows to check for constraint
    AnalyzeResult(RouteMatrix, NumLocations)                                                                                  # Check if RouteMatrix satisfies other constraints as well

    #############################################################################################
    ###### Calculate the total cost of the route the spaceship made (in millions of of km - distance)
    TotalRouteCost = 0
    for trips in range(0, NumLocations):
        TotalRouteCost = TotalRouteCost+float(CostMatrix.item(RouteMatrixElements[trips], RouteMatrixElements[trips + 1]))     # The sum of the matrix elements where the spaceship has been (determined through the indices)
    RouteDict['RouteCost'] = {'Cost':TotalRouteCost}

    ##### Return the simulation result in a human understandable way =)
    return RouteDict

############################################################################################
##### Check whether the solution satisfies the constraints 
def AnalyzeResult(RouteMatrix, NumLocations):

    print(RouteMatrix) # print the route to the terminal so that you can debug/re-weight the constraints

    ############################################################################################                        
    ##### Check if the number of travels is equal to the number of locations (+ 1 for returning home)
    if (len(RouteMatrix) - 1) != NumLocations + 1:
        raise RuntimeError('This solution is not valid -- Number of locations visited invalid!')
    else:
        NumLocationsPassed = NumLocations 
        print(f"Number of planets, moons, and asteroids passed = {NumLocationsPassed}. This is correct!")

    ############################################################################################                        
    ##### Check if the locations are different (except start/end location)
    PastLocations = []
    for k in range(1, len(RouteMatrix) - 1):                                                                           # Start to second last location must all be different - skip header so start at 1, skip last location so - 1
        for l in range(0, len(PastLocations)):  
            if RouteMatrix[k][1] == PastLocations[l]:
                raise RuntimeError('This route is not valid -- Traveled to a non-starting planet/moon/or asteroid more than once')
        PastLocations.append(RouteMatrix[k][1])
    print(f"Number of different planets, moons, asteroids passed = {NumLocations}. This is correct!") 

    ############################################################################################                        
    ##### Check if the end location is same as the start location
    if RouteMatrix[1][1] != RouteMatrix[-1][1]:
        raise RuntimeError(f'This route is not valid -- Start ({RouteMatrix[1][1]}) and end ({RouteMatrix[-1][1]}) location are not the same!')
    print('Start and end location are the same. This is correct!')

    ############################################################################################                        
    ##### Check if start location and end location are correct
    if RouteMatrix[1][1] != 'Mars' or RouteMatrix[-1][1] != 'Mars':
        raise RuntimeError(f'This solution is not correct -- Start location {Path[1][1]} or end location {Path[-1][1]} incorrect')
    print('Start and end location are as planned. This is correct!')

    print('Valid route!')

############################################################################################                        
##### Define the solver --- uncomment if you wish to use a different one --- you can also change the timeout.

solver = SimulatedAnnealing(workspace, timeout = 120)   
#solver = ParallelTempering(workspace, timeout = 120)
#solver = Tabu(workspace, timeout = 120)
#solver = QuantumMonteCarlo(workspace, sweeps = 2, trotter_number = 10, restarts = 72, seed = 22, beta_start = 0.1, transverse_field_start = 10, transverse_field_stop = 0.1) #QMC is not available parameter-free yet, you will need to tune these solver parameters. 

############################################################################################                        
##### Submit the optimization problem to the Azure Quantum solvers --- uncomment if you wish to use a different one --- you can also change the timeout.
OptimizationProblem = OptProblem(CostMatrix)
route = solver.optimize(OptimizationProblem)     # Solve the optimization problem as a synchronous job -- wait until done.

PathDict = ReadResults(route['configuration'], LocationNames, CostMatrix, NumLocations)
print(PathDict)
```

## A working navigation system

Congratulations, you have finished building the navigation system for the spaceship! With the routes provided by the navigation system there is no doubt that the mining expeditions will be succesful. To give an impression of how difficult this problem would have been to solve by hand, it would require writing out 362,880 routes. A lot of effort saved! You can now also try adding more planets, moon, and asteroids to the itinerary to see how the routes and difficulty of the problem change. 





