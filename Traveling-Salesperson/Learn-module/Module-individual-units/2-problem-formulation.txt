For the module, we define the traveling salesperson problem for the spaceship as follows: 

You have a set of $N$ locations (planets, moons, and asteroids) that need to be visited. As the newly appointed navigation engineer, you have to code the navigation system in order to find the shortest routes through the solar system. 

There are a number of conditions that you will need to fullfill in order for the navigation system to be considered 'successful' by the crew. These will be considered the problem constraints:

- Conditions:
  - `One location at a time constraint` - The spaceship can only be in one planet, moon, or asteroid at a time. No magic!
  - `No dissapearing constraint` - The spaceship has to be in a location, it can not disappear!
  - `No revisiting locations constraint` - The spaceship may **not** visit a location more than once, except the homebase (start/end location). 
  - `Start and end locations constraints` - The spaceship has to start and finish in the home base, which will be Mars.

- Please consider the following points for the module as well:
  - We will ignore orbital mechanics and use real approximate mean distances between the planets, moons, and asteroids. Distances are in millions of kilometers. All locations have a terrestrial surface (no gas planet included, as that is hard to mine).
  - We will not specify where the space rocks can, and can not be, sold and mined. However, you may expand the constraints yourself to include such criteria.
  - The $N$ locations are (in order) : Mars, Earth, Earth's Moon, Venus, Mercury, Ganymede (moon), Titan (moon), Ceres (asteroid), Pallas (asteroid), Cybele (asteroid). Naming the planets will make it easier to read the solutions returned by the solver!
  

## About the problem
The goal behind reprogramming the navigation system is to find a suitable interplanetary route to minimize the spaceship's travel distances. Having a (near)-optimal route will not only lower the total distance but will also reduce the ship's energy consumption and the number of working hours for the crew. An efficient route is crucial to the overall productivity of the spaceship's mining ambitions!

The main idea is to create a **cost function**, which is used to model the travel costs **and** travel constraints of the spaceship. A particular route wants to be found that minimizes the cost function as much as possible. By incorporating the constraints into the cost function through penalties, the solver can be encouraged to settle for certain solution sets. Intuitively, you can imagine this as designing the rugged optimization landscape such that the 'good' solutions (routes) have lower cost values (minima) than 'bad' solutions. Even though the problem sounds easy to solve, for a large number of locations it becomes nearly impossible to find the optimal route (global minima). The difficulty is predominantly caused by:
- the rugged optimization landscape (non-convex). 
- the combinatorial explosion of the solution space, meaning that you may not be able to check every possible route.
- the variables optimized for are non-continuous, $x_{v} \in \{0,1\}.$

> [!TECHNICAL NOTES]
> The traveling salesperson problem in which an optimal route needs to be found belongs to the set of [NP-hard problems](https://en.wikipedia.org/wiki/NP-hardness?azure-portal=true). First, there is no explicit algorithm for which you can find the optimal solution, meaning that you are faced with a search of the $(N-1)!$ (factorial number of routes to pass through all the locations) possible routes. Secondly, to verify whether a candidate solution to the traveling salesperson problem is optimal, you would have to compare it to all $(N-1)!$ other candidate solutions. Such a procedure would force you to compute all routes which is an **extremely** difficult task for large $N$ (non-polynomial time $\mathcal{O}((N-1)!)$).
> The total number of routes for the traveling salesperson problem is dependent on the problem formulation. For generality, we will consider the directed version (`directed graph`) of the traveling salesperson problem, meaning that every route through the network is unique. Therefore you can assume that there are $(N-1)!$ different routes possible, since a starting location is given. 

## Azure Quantum workspace

As presented in the previous two modules, [Solve optimization problems by using quantum-inspired optimization](/learn/modules/solve-quantum-inspired-optimization-problems?azure-portal=true) and [Solve a job scheduling optimization problem by using Azure Quantum](/learn/modules/solve-job-shop-optimization-azure-quantum?azure-portal=true), the optimization problem can be submitted to Azure Quantum solvers. For this, we will use the the Python SDK and format the traveling salesperson problem for the solver with cost function `terms`. 
In order to submit the optimization problem to the Azure Quantum solver later, you will need to have an Azure Quantum workspace. Follow these [directions](/learn/modules/get-started-azure-quantum?azure-portal=true) to set one up if you don't have one already.

Create a Python file or Jupyter Notebook, and be sure to fill in the details below to connect to the Azure Quantum solvers and import relevant Python modules. In case you don't recall your `subscription_id` or `resource_group`, you can find your Workspace's information on the Azure Portal page.
Throughtout the module we will be appending to the Python script below such that you can run and view intermediary results. 

```python

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
```
The first time you run this code on your device, a window might prompt in your default browser asking for your credentials.

## Definitions

A list to quickly look up variables and definitions:  

- Locations: The planets, moons, and asteroids. 
- Origin: A location the starship departs from.
- Destination: A location the starship travels to. 
- Trip: A travel between two locations. For example, traveling from Mars to Earth.
- Route: Defined as multiple trips. Traveling between more than two locations. For example, Mars &rarr; Earth &rarr; Celeste &rarr; Venus.
- $N$: The total number of locations. 
- $C$: The travel cost matrix. In this module, the costs are distances given in millions of kilometers. 
- $i$: Variable used to index the rows of the cost matrix, which represent origin locations.
- $j$: Variable used to index the columns of the cost matrix, which represent destination locations. 
- $c_{i,j}$: The elements of the matrix, which represent the travel cost, that is the distance, between origin $i$ and destination $j$.
- $x_v$: The elements of a location vector. Each represents a location before or after a trip. These are the optimization variables.
- $w_b$: Constraint weights in the cost function. These need to be tuned to find suitable solutions.

## Problem formulation

With problem background dealt with, you can start looking at modeling it. In this unit, we will first go over how to calculate the spaceship's travel costs. This will become the foundation of the cost function. In later units, you will expand the cost function by incorporating penalty functions (constraints), to find more suitable routes throughout the solar system.

Let's get to work!

### Defining the travel cost matrix $C$

Consider a single trip for the spaceship, from one planet to another. The first step is to give each location (planet, moon, asteroid) a unique integer in $\{0,N-1\}$ (N in total). Traveling from planet $i$ to planet $j$ will then have a travel distance
of $c_{i,j}$, where $i$ denotes the `origin` and $j$ the `destination`.

$$ \text{The origin location } (\text{node}_i) \text{ with } i \in \{0,N-1\}.$$
$$ \text{The destination location } (\text{node}_j) \text{ with } j \in \{0,N-1\}.$$
$$ \text{Distance from } \text{location}_i \text{ to } \text{location}_j \text{ is } c_{i,j}.$$

Writing out the travel distances for every $i$-$j$ (origin-destination) combination gives the travel cost matrix $C$:

Travel cost matrix:
$$C = \begin{bmatrix} c_{0,0} & c_{0,1} & \dots & c_{0,N-1} \\ c_{1,0} & c_{1,1} & \dots & c_{1,N-1} \\ \vdots & \ddots & \ddots & \vdots \\ c_{N-1,0} & c_{N-1,1} & \dots & c_{N-1,N-1} \end{bmatrix}. $$

Here, the rows are indexed by $i$ and represent the origin locations. The columns are indexed by $j$ and represent the destination locations. For example, traveling from location $0$ to location $1$ is described by:

$$ C(0,1) = c_{0,1}.$$ 

The travel cost matrix is your spaceflight "dictionary", the most important asset to you as the spaceship's navigation engineer. It contains the most crucial information in finding a suitable route through the solar system to sell and mine space rocks. The measure of the travel cost is arbitrary. In this module the cost we want to minimize is the distance, however you can also use time, space debris, space money, ..., or a combination of different costs. 

### Defining the location vectors

With a travel cost formulation between two locations complete, a representation of the origin and destination needs to be defined to select an element of the travel cost matrix. Remember, this is still for a single trip between location $i$ and location $j$. Selecting an element of the matrix can be achieved by multiplying the matrix with a vector from the left, and from the right! The left vector and right vector specify the origin and the destination, respectively. Consider the example where the spaceship travels from planet $0$ to planet $1$: 

$$ \text{Travel cost - location 0 to location 1 }=  \begin{bmatrix} 1 & 0 & \dots & 0 \end{bmatrix} \begin{bmatrix} c_{0,0} & c_{0,1} & \dots & c_{0,N-1} \\ c_{1,0} & c_{1,1} & \dots & c_{1,N-1} \\ \vdots & \ddots & \ddots & \vdots \\ c_{N-1,0} & c_{N-1,1} & \dots & c_{N-1,N-1} \end{bmatrix} \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix} = c_{0,1}.$$

Recall that the spaceship can only be in one location at a time, and that there is only one spaceship, therefore only one element in the origin and destination vectors can equal 1, while the rest of them are 0. In other words, the sum of elements of the origin and destination vectors must equal 1. We will work this into a constraint in a later unit.

Fantastic, you now know how to extract information from the travel cost matrix $C$ and express the travel cost for a single trip. However, it is necessary to express these ideas in a mathematical format for the solver. In the previous example the trip was hard-coded to go from location $0$ to location $1$. Let's generalize for any trip:

$$ x_v \in \{0,1\} \text{ for } v \in \{0,2N-1\}, $$

$$ \text{Travel cost for single trip }=  \begin{bmatrix} x_0 & x_1 & \dots & x_{N-1} \end{bmatrix} \begin{bmatrix} c_{0,0} & c_{0,1} & \dots & c_{0,N-1} \\ c_{1,0} & c_{1,1} & \dots & c_{1,N-1} \\ \vdots & \ddots & \ddots & \vdots \\ c_{N-1,0} & c_{N-1,1} & \dots & c_{N-1,N-1} \end{bmatrix} \begin{bmatrix} x_{N} \\ x_{N+1}\\ \vdots \\ x_{2N-1} \end{bmatrix}. $$

Each $x_v$ represents a location before or after a trip. The reason that the destination vector elements are indexed from $N+1$ to $2N$ instead of $0$ to $N-1$ is to differentiate the origin variables and destination variables for the solver. If this is not done, the solver would consider the origin and destination the same, meaning that the spaceship would never take off to its destination planet! When submitting the optimization problem, the solver will determine which $x_v$ is assigned a value 1 or 0, dependent on the respective trip's travel cost. To re-iterate an important point, the sum of the origin and destination vector elements must be 1 as the spaceship can not be in more than or less than one location at a time: 

$$ \text{Sum of the origin vector elements: }\sum_{v = 0}^{N-1} x_v = 1, $$ 
$$ \text{Sum of the destination vector elements: }\sum_{v = N}^{2N-1} x_v = 1.$$ 


Now that the basic mathematical formulations are covered, the scope can be expanded. As the spaceship's navigation engineer, it is your job to calculate a route between multiple planets, moons, and asteroids, to help make the space rock mining endeavour more succesful. With the travel cost matrix and the origin/destination vectors defined, you are equipped with the right mathematical tools to start looking at routes through the solar system. Let's take a look at how two trips can be modeled!


### Defining the travel costs for a route

To derive the travel costs for two trips, which can be considered a route, you will need a way to describe the **total** travel cost. As you might expect, the total cost of a route is simply the sum of the trip's travel costs that constitute the route (sum of the trips). Say you have a route $R$ in which the spacehip travels from location $1$ to location $3$ to location $2$. Then the total cost is the sum of the trips' costs:

$$ \text{Cost of route: } R_{1-3-2} = c_{1,3}+c_{3,2}. $$

By taking a closer look at the equation a very important relation is found. The destination for the first trip and the origin for the second trip are the same. Incorporating this relation in the cost function will help reduce the number of variables the solver has to optimize for. As a result of the simplification, the optimization problem becomes much easier and has an increased probability of finding suitable solutions! 

> [!TECHNICAL NOTE]
> For the 2-trip example this recurrence relation reduces the solver search space size (number of possible routes/solutions) from $2^N \cdot 2^N \cdot 2^N \cdot 2^N$ to $2^N \cdot 2^N \cdot 2^N$, a factor $N$ ($N$ = 10 in this module) difference. Without considering constraints, each vector element can take two values (${0,1}$) and has a length $N$, therefore each vector multiplies the solution set by $2^N$. The reduction in variables becomes even more apparent for longer routes. Visiting the 10 planets, moons, and asteroids as in this module, the relation reduces the search space size from $2^{20N}$ to $2^{11N}$! The number of variables optimized for decreases from $200$ to $110$, respectively.

Recall that the travel costs can be written with vectors and matrices. Then the total travel cost of the route is: 

$$ \text{Cost of route } = \begin{bmatrix} x_0 & x_1 & \dots & x_{N-1} \end{bmatrix} \begin{bmatrix} c_{0,0} & c_{0,1} & \dots & c_{0,N-1} \\ c_{1,0} & c_{1,1} & \dots & c_{1,N-1} \\ \vdots & \ddots & \ddots & \vdots \\ c_{N-1,0} & c_{N-1,1} & \dots & c_{N-1,N-1} \end{bmatrix} \begin{bmatrix} x_{N} \\ x_{N+1}\\ \vdots \\ x_{2N-1} \end{bmatrix} + \begin{bmatrix} x_{N} & x_{N+1} & \dots & x_{2N-1} \end{bmatrix} \begin{bmatrix} c_{0,0} & c_{0,1} & \dots & c_{0,N-1} \\ c_{1,0} & c_{1,1} & \dots & c_{1,N-1} \\ \vdots & \ddots & \ddots & \vdots \\ c_{N-1,0} & c_{N-1,1} & \dots & c_{N-1,N-1} \end{bmatrix} \begin{bmatrix} x_{2N} \\ x_{2N+1}\\ \vdots \\ x_{3N-1} \end{bmatrix}.$$

In the equation the destination vector of the first trip and the origin vector of the second trip contain the same variables, making use of the recurrence relation. Generalizing the 2-trip route to a $N$-trip route is achieved by including more vector-matrix multiplications in the addition. Written out as a sum:

$$\text{Travel cost of route } = \sum_{k=0}^{N-1} \left(  \begin{bmatrix} x_{Nk} & x_{Nk+1} & \dots & x_{Nk+N-1} \end{bmatrix} \begin{bmatrix} c_{0,0} & c_{0,1} & \dots & c_{0,N-1} \\ c_{1,0} & c_{1,1} & \dots & c_{1,N-1} \\ \vdots & \ddots & \ddots & \vdots \\ c_{N-1,0} & c_{N-1,1} & \dots & c_{N-1,N-1} \end{bmatrix} \begin{bmatrix} x_{N(k+1)} \\ x_{N(k+1)+1}\\ \vdots \\ x_{N(k+1)+N-1} \end{bmatrix} \right),$$

which can equivalently be written as:
 
$$\text{Travel cost of route} = \sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right),$$

where the $x$ variables indices are dependent on the trip number $k$, the total number of locations $N$, and the origin ($i$) and destination ($j$) locations. 

Great! A function to calculate the travel costs for the spaceship's route has been found! Because you want to minimize (denoted by the 'min') the total travel cost with respect to the variables $x_v$ (written underneath the 'min'), we will write the foundation of the cost function as follows:

$$\text{Travel cost of route} := \underset{x_0, x_1,\dots,x_{(N^2+2N)}}{min}\sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right).$$

This equation will not make up the entire cost function for the solver. Penalty functions have to added to it for the constraints, otherwise the solver will return invalid solutions. These will be added to the cost function in upcoming units. 

The crew is very excited with the progress you have already booked. You are ready to program the travel costs into navigation system you are building for the spaceship. Time to write some code!


### Progress on the navigation system's cost function

- The cost function contains:
  - The `travel costs`.

$$ \text{Cost function so far: } \underset{x_0, x_1,\dots,x_{(N^2+2N)}}{min} \left(\sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right)\right) $$

### Coding the travel costs 

For the solvers to find a suitable solution, you will need to specify how it should calculate the travel cost for a route. The solver requires you to define a cost term for each possible trip-origin-destination combination given by the variables $k$, $i$, $j$, respectively. As described by the cost function above, the weighting is simply the $c_{i,j}$-th element of the cost matrix, resembling the distance between the locations. For example, weighting a trip from location $1$ to location $2$ in the second trip ($k=1$) has the following weighting:

$$ c_{i,j} \cdot x_{Nk+i} \cdot x_{N(k+1)+j} =  c_{1,2} \cdot x_{N+1} \cdot x_{N+2}. $$

The $x_v$ variables have an $N$ term because for the solver we need to differentiate between the variables over the trips. In other words, after each trip there are $N$ new variables that represent where the spaceship can travel to next.

Below, you can find the code snippet that will be added to the Python script. We define a problem instance through the `OptProblem` function, in which we will continue to append pieces of code throughout the module's units. Later, this problem will be submitted to the Azure solvers. If you want to see how the weights are assigned to each $ x_{Nk+i} \cdot x_{N(k+1)+j}$ combination, you can uncomment the last lines in the code. Lastly, since the optimization variables $x_v$ can take values ${0,1}$, the problem type falls into the `pubo` category (polynomial unconstrained binary optimization). 

```python

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
                #print(f'For node_{i}, to node_{j} in trip number {k} costs: {CostMatrix.item((i,j))}')                                         # In a format that is easier to read for a human

    return Problem(name="Spaceship navigation system", problem_type=ProblemType.pubo, terms=terms)

OptimizationProblem = OptProblem(CostMatrix)

``` 


## Next Steps

As navigation engineer of the spaceship you have completed the initial work on the route planner. But with the current code, the Azure Quantum solvers on Earth would send routes to the spaceship that would make no sense! Currently, all $x_v$ would be assigned a value 0 by the solver because that would yield a value-0 travel cost! To avoid this from happening, you will need to implement the constraints into the cost function. These constraints will penalize the routes that are incorrect. More on this in the next units! 

