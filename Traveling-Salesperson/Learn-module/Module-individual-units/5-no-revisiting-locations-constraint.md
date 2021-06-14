In this unit, you will be making sure that the spaceship does not visit any planet, moon, or asteroid more than once (except the home base). Space rocks are crucial to the construction and development of the newest scientific gadgets. Therefore a key part of mining expidition is also selling these rocks to the science and astronaut communities in our solar system. On each expedition, you want to visit each location to sell and mine the space rocks. The *"no revisiting locations"* constraint will penalize routes that revisit any planets, moons, and asteroids. 

> Currently, the solver may provide solutions in which the spaceship revisits a location. For example, there is no cost associated with staying in the same location, as shown by the diagonal elements of the cost matrix. It is necessary to penalize routes in which revisits occur. 


## "No revisiting locations" constraint

The navigation system may generate routes in which a location is visited more than once. Similarly to previous modules, we will have to penalize routes in which this occurs. To start, let's consider the fourth location $x_3$ (Venus) for each trip:

$$ x_3, x_{3+N}, x_{3+2N}, ... $$

For a valid route, the spaceship will only have passed through this location once. That means that only one of these variables is allowed to take a value 1, the remaining have to be 0. For example, if the spaceship visits Venus after the first trip, then $x_{3+N}=1$ and $x_3=0$, x_{3+2N}=0$, and so on. As done similarly with the one location at a time constraint, the product of these variables must equal 0. The following can then be derived:

$$ x_3 \cdot x_{3+N} \cdot x_{3+2N} \cdot ... = 0. $$

Even though this constraint might seem correct, it is not stringent enough. If only one of these variables is 0, then the constraint is already satisfied. However, the multiplication of variables can be split into their individual products:

$$ x_3 \cdot x_{3+N} =0,$$
$$ x_3 \cdot x_{3+2N} = 0,$$
$$ x_{3+N} \cdot x_{3+2N} = 0,$$
$$ \vdots $$

By weighting these terms in the cost function, at least one of the two variables for each respective equation has to be zero. The solver will therefore be penalized if more than one variable takes the value 1. Continuing this relation for all $N$ trips and all $N$ locations yields the penalty function:

$$ 0 = \large{\sum}_{p=0}^{N^2+N-1}\hspace{0.25cm} \large{\sum}_{f=p+N,\hspace{0.2cm} \text{stepsize: } N}^{N^2-1} \hspace{0.35cm} (x_p \cdot x_f),$$

in which the first summation ($p$) assigns a reference location $x_p$, and the second summation assigns the same location but after a number of trips $x_f$ (multiple of $N$ due to the stepsize). 


### Progress on the navigation system's cost function

- The cost function contains:
  - The `travel costs`.
  - The `one location at a time constraint`, with constraint weight $w_1$. 
  - The `no dissapearing constraint`, with constraint weight $w_2$.
  - The `no revisiting locations constraint`, with constraint weight $w_3$.

$$ \text{Cost function so far: } \underset{x_0, x_1,\dots,x_{(N^2+2N)}}{min} \left(\sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right)  +   w_1 \left( \sum_{l=0}^{N} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} x_{(i+Nl)} \cdot x_{(j+Nl)} \text{ with } \{ i,j | i<j \} \right) +w_2\left( \sum_{k=0}^{N(N+1)-1} x_k \right) + w_3 \left(  \large{\sum}_{p=0}^{N^2+N-1}\hspace{0.25cm} \large{\sum}_{f=p+N,\hspace{0.2cm} \text{stepsize: } N}^{N^2-1} \hspace{0.35cm} (x_p \cdot x_f)   \right)          \right) $$


## Coding the constraint
Integrating this into the cost function will penalize routes in which locations are visited more than once. The derived model does not 
penalize the last trip in which the spaceship should return to the home base. The reason is that the last trip ($N$-th trip) would always violate the constraint, as the $N$ different locations have all already been visited. Also, adding it would only make the cost function larger without adding any value to the optimization problem. In the next unit, you will look at how the spaceship can end at a specific location.


``` python

    ############################################################################################                        
    ##### Constraint: no revisiting locations constraint --- (in the last step we can travel without penalties (this is to make it easier to specify an end location ))
    for p in range(0,len(CostMatrix)+len(CostMatrix)*(len(CostMatrix))):                                  # This selects a present location x: 'p' for present    
        for f in range(p+len(CostMatrix),len(CostMatrix)*(len(CostMatrix)),len(CostMatrix)):              # This selects the same location x but after upcoming trips: 'f' for future
            terms.append(
                Term(
                    c =int(0),          
                    indices = [p,f]   
                )
            )     
            ##----- Uncomment one of the below statements if you want to see how the weights are assigned! -------------------------------------------------------------------------------------------------
            #print(f'x_{p},x_{f}')  # Just variable numbers 
            #print(f'Visit once constraint: x_{p} - x_{f}  assigned weight: 0')  # In a format for the solver (as formulated in the cost function)
            #print(f' Visit once constraint: location_{p%NumLocations} - location_{(p+f)%NumLocations} after {(f-p)/NumLocations} trips assigned weight: 0')  # In a format that is easier to read for a human


    return Problem(name="Spaceship navigation system", problem_type=ProblemType.pubo, terms=terms)

OptimizationProblem = OptProblem(CostMatrix)  


``` 

## Next steps

Wow! The navigation system is nearly complete. Shortly, you will be able to assist the crew in their logistics and planning departments! There are still some final steps to complete before you are ready to demonstrate it. You still need a way to include some start and end locations and complete the weight tuning steps. In the next unit, the start and end locations will be added to the cost function. 
