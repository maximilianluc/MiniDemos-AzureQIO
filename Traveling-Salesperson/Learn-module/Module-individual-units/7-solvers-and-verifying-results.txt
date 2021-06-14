The minimization problem is ready to be submitted to the solvers in the Earth's Azure Quantum workspace. But before that, the navigation system now requires some fine-tuning and solver selection. In this unit you will learn how to read and analyze solutions returned by the solver, and about some differences in the available solvers. Make sure the crew buckles up, interplanetary routes will be available soon!

## Some notes about solvers

There are numerous solvers (algorithms) and targets (hardware providers) available in the Azure Quantum optimization service. Choosing a which to use is not straightforward, it is recommended to review the [Microsoft QIO provider](/azure/quantum/provider-microsoft-qio?azure-portal=true) and [1QBit provider](/azure/quantum/provider-1qbit?azure-portal=true) documentation pages for further information and references. We will briefly cover some background of the solvers which might help you decide how you would like to submit the optimization problem. 

Unfortunately, there is no standardized way of finding the best solver for (highly) non-convex problems. For these optimization problems you are left with two options. First, you can use `heuristic solvers`. Heuristic methods can be understood as searches of the optimization landscape in which as many minima want to be found as possible. The second option is to use linearizations of the problem, suitable for small and 'easy' non-convex problems (`constraint relaxation`, `cost function linearization`, etc.). For linearizations, a degree of accuracy is paid to reduce the problem complexity. However, for large non-convex problems linearizations usually perform poorly are as they generate **local** and **approximate** solutions. (Approximating nonlinearies with linear functions becomes increasingly difficult with the degree of nonlinearities). 

For the traveling salesperson problem, it is practically required to use heuristic solvers due to the combinatioral explosion with the number of locations. The main bottleneck is how to efficiently design/weight the cost function and how to choose the right solver. Choosing an appropriate solver requires some research as each heuristic method is tailored for specific problem conditions. It can therefore be that a solver performs poorly on a certain problem instance, but well on another. Usually, finding a suitable one requires experimentation, and/or more thorough insight into the problem. To help you with that, you can consider the following notes:

1. What are the problem properties? 
  - Is the problem convex or non-convex (rugged)? Is the optimization landscape slightly rugged(hills), or very rugged (peaks)?
  - How large is the optimization problem (given by the number of variables $x_v$)?  
  - Are you looking at an Ising model ($x_v \in \{-1,1\}$ ), or a PUBO/QUBO model ($x_v \in \{0,1\}$)?
  - Can you reduce or simplify the optimization problem? Smaller cost functions will reduce computational effort for the solvers, and save time!
2. Finding solvers through experimentation. 
  - Try a solver that you can use as a reference to compare to other solver performances, such as accuracy or time. For example, simulated annealing generally does well on a wide range of problems and can provide an initial metric to evaluate other solvers.
  - Also, in early stages it can be helpful to use a solver for which parameters do not need to be tuned (`parameter-free`).
  - Select solvers which match well with the problem, for an overview see the documentation [solvers and targets](/azure/quantum/qio-target-list?azure-portal=true).  
  - Try other solvers and compare their performances. 
  - Do all solvers perform poorly? Consider redesigning the cost function or a longer solver time (`timeout`). Often you can make the cost function more efficient for the solver, some tips are described in the [documentation](/azure/quantum/provider-microsoft-qio?azure-portal=true).
  - Are there some good results? Consider these solvers as candidates for the fine-tuning process. Fine-tuning can be a lot of work, therefore it is useful to have a small subset of solvers. 
  - Solver selection is an iterative process. If possible, it can be helpful to experiment on small-scale problem as this might be more insightful and reduce effort. 
3. Fine-tuning candidate solvers.
  - Improve the performances by defining solver parameters such as, start conditions, convergence criteria, sweeps, etc. 
  - Depending on the scale of the problem and solver type, you can consider different hardware (CPUs / FPGAs / GPUs / Annealers) and targets. 
  - Adjust the cost function weights specifically for a type of solver.
  - Solver tuning is also an iterative process.

## Submitting the optimization problem to Azure Quantum

Submitting the navigation system's cost function to the Azure solvers will require a solver. As navigation engineer, it is important to know advantages and disadvantages of these methods, and for testing reasons, you want to give them all a go. 

Below, you can find the code snippet that will be added to the python script. The solver hardware is submitted to the Microsoft QIO target and defaulted to CPU hardware. The 120 seconds for `timeout` defines when the optimization procedure should terminate (approximately). The four available solvers are `simulated annealing`, `parallel tempering`, `tabu search`, and `quantum Monte Carlo`. The first three are available parameter-free, meaning that they are tuned/inferred for you by the Azure Quantum solvers. For the `quantum Monte Carlo` algorithm this is not yet available, and therefore it has some necessary parameters predefined. 

``` python

############################################################################################                        
##### Define the solver --- uncomment if you wish to use a different one --- you can also change the timeout.

solver = SimulatedAnnealing(workspace, timeout = 120)   
#solver = ParallelTempering(workspace, timeout = 120)
#solver = Tabu(workspace, timeout = 120)
#solver = QuantumMonteCarlo(workspace, sweeps = 2, trotter_number = 10, restarts = 72, seed = 22, beta_start = 0.1, transverse_field_start = 10, transverse_field_stop = 0.1) # QMC is not available parameter-free yet


route = solver.optimize(OptimizationProblem)     # Solve the optimization problem -- wait until done.
print(route)


```

## Reading and analyzing the results

You should always verify the solution returned by the solver. It could be that the solver converged to an unsuitable minima, reflected by the solution configuration. By checking if there are constraint violations and the solution's cost value, you can determine the validity of a solution. By adding such a verification procedure in the code, the constraint weight tuning will be more insightful. Also, the navigation system becomes much more fault tolerant! 

The verification procedure consists of the two functions defined below. The first, `ReadResults`, converts the returned solution (`Config`) into a more readable format and calculates the cost of the route. The second, `AnalyzeResult`, assesses whether the route conforms with the problem constraints. 

```python

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

```

## Next steps

In the next unit, you will add the code pieces together and submit the problem to the Azure solvers. By tuning the constraint weights you will improve the routes for the spaceship. By the end, you should have a working navigation system ready for the crew!


