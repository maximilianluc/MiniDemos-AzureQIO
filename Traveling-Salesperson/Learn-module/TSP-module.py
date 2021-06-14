### FILL IN THE CONSTRAINT WEIGHTS AND SUBSCRIPTION DETAILS TO SUBMIT!

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
