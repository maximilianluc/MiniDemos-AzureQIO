With the foundations of the navigation system in place, it is time to incorporate the constraints. The initial results made the crew enthousiastic! However, before launching the final system, you will need to implement penalty functions to ensure that the Azure Quantum solvers generate feasible routes through the solar system. 

In this unit we will consider the *"one location at a time"* constraint. Recall that the spaceship can only be in one location at a time. It is impossible for the spasceship to be on both Earth and Mars simultaneously! You will need to program this into the navigation system by penalizing the solvers for routes in which such magic happens. 

>[!Note]
>From this unit onwards, the origin and destination vectors will be referred to as **location vectors** due to the recurrence relation presented in the previous unit. The location vector is used to represent the location of the spaceship after a number of trips. For example, starting from the home base will give the location vector 0. After completing the first trip, the spaceship will be at a location given by location vector 1. 


## "One location at a time" constraint

The spaceship can only be at one planet, moon, or asteroid at a time. So far, the defined cost function only contains information about the travel costs, nothing about whether the spaceship is at multiple locations at once. To mathematically express this idea, it must be true that only one element in a location vector equals 1. For example:

$$ \text{Incorrect location vector: } \begin{bmatrix} 1 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, $$

should not be allowed as the spaceship is at the first two locations simultanesouly. Only one element may be non-zero:

$$ \text{Correct location vector: } \begin{bmatrix} 0 \\ 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}. $$

One way of designing this constraint would be look at the sum of the location vector elements, which must equal 1: 

$$ \text{Location vector 0: (home base)} \hspace{0.5cm}  x_0 + x_1 + \dots + x_{N-1} = 1, $$
$$ \text{Location vector 1: } \hspace{0.5cm} x_{N} + x_{N+1} + \dots + x_{2N-1} = 1, $$
$$ \text{Location vector N (home base): } \hspace{0.5cm} x_{N^2} + x_{N^2+1} + \dots + x_{N^2 + N-1} = 1. $$

Enforcing this constraint over all trips would then give (N+1 because the spaceship returns to the home base on Mars):

$$ \text{For all locations: } \hspace{0.5cm}  x_0 + x_1 + \dots +  x_{N^2 + N-1} = N+1. $$

This equation is a valid way to model the constraint. However, there is a drawback to it as well. In the equation, only the individual locations are penalized. There is no information about being in two locatoins at once! The following values satisfy the equation:

$$ \text{If } x_0=1, x_{1}=1, \dots, x_{N}=1, \text{ and the remaining } x_{N+1}=0, x_{N+3}=0, \dots, x_{N^2+N-1}=0,$$  

but violate the constraint we are trying to model. With these values, the spaceship is at all planets/moons/asteroids at once before the first trip (location vector 0). Let's rethink. Consider a small example of three locations (a length-3 location vector). Then if the spaceship is in location 0, it can not be in location $1$ or $2$. Instead of using a summation to express this, another valid way would be to use products. The product of elements of a location vector must always equal zero, regardless where the spaceship is, because only one of the three $x_v$ elements can take value 1. Writing out the products of a single location vector gives:

$$ x_0 \cdot x_1 = 0,$$
$$ x_0 \cdot x_2 = 0,$$ 
$$ x_1 \cdot x_2 = 0.$$

In this format, the constraint is much more specific and stringent for the solver. These equations reflect the interrelationships between the locations more accurately than the summation. They can be implemented in a way that distinguish the variables for different location vectors, unlike the summation which adds all the variables of the location vectors together. As a result of describing the constraint more specifically, the solver will return better solutions. 

>[!Note]
>We do not want to weight combinations between locations more than once, as this would lead to inbalances (assymmetries) in the cost function. Tuning the weights of an imbalanced cost function tends to be more difficult. We therefore exclude the reverse combinations:
>$$ x_{1}\cdot x_{0}, $$
>$$ x_{2}\cdot x_{0}, $$
>$$ x_{2}\cdot x_{1}. $$

The next step consists on generalizing the one location at a time constraint for the spaceship, which has to pass by all locations in the solar system and return to home base (N+1 locations need to be visited). In the equation below, $l$ iterates over the number of location vectors, while $i$ and $j$ iterate over the vector elements:

$$ \text{One location at a time constraint: }0=\sum_{l=0}^{N} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} x_{(i+Nl)} \cdot x_{(j+Nl)} \text{ with } \{ i,j | i<j \} $$


### Progress on the navigation system's cost function

- The cost function contains:
  - The `travel costs`.
  - The `one location at a time constraint`, with constraint weight $w_1$. 

$$ \text{Cost function so far: } \underset{x_0, x_1,\dots,x_{(N^2+2N)}}{min} \left(\sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right)  +   w_1 \left( \sum_{l=0}^{N} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} x_{(i+Nl)} \cdot x_{(j+Nl)} \text{ with } \{ i,j | i<j \} \right) \right) $$

## Coding the constraint

Great! With the mathematics written out, it is time to add the constraint into the navigation system's code. As in previous modules, you'll need to assign constraint weights to penalize invalid solutions. For now, you can ignore them as they need to be tuned after coding the remaining constraints. Weight tuning will require a bit hands-on experimentation, as they will impact the satisfiability of the route returned by the solver.


``` python

    ############################################################################################
    ##### Constraint: One location at a time constraint - spaceship can only be in 1 location at a time.
    for l in range(0,len(CostMatrix)+1):                # The total number of locations that are visited over the route (N+1 because returning to home base)
        for i in range(0,len(CostMatrix)):              # For each location (iterate over the location vector)
            for j in range(0,len(CostMatrix)):          # For each location (iterate over the location vector)
                if i!=j and i<j:                        # i<j because we don't want to penalize twice // i==j is forbidden (this could equal 1, that's why)
                    terms.append(
                        Term(
                            c = int(0),                                    
                            indices = [i+(len(CostMatrix)*l),j+(len(CostMatrix)*l)]                   
                        )
                    )
                    ##----- Uncomment one of the below statements if you want to see how the weights are assigned! -------------------------------------------------------------------------------------------------
                    #print(f'{i+(len(CostMatrix)*l)},{j+(len(CostMatrix)*(l))}')
                    #print(f'Location constraint 1: x_{i+(len(CostMatrix)*l)} - x_{j+(len(CostMatrix)*(l+1))} (trip {l}) assigned weight: 0')  # In a format for the solver (as formulated in the cost function)


    return Problem(name="Spaceship navigation system", problem_type=ProblemType.pubo, terms=terms)

OptimizationProblem = OptProblem(CostMatrix)  


``` 


## Next steps

The first constraint has been finished. Significant progress is being made on the spaceship's navigation system! Let's keep going and work on the implementing the second constraint, the spaceship may not disappear in the navigation system! 



