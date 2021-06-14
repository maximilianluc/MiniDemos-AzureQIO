The crew of the spaceship has decided to start and end the mining expedition on their home base on Mars. You will need to code these conditions into the navigation system. Fortunately, relative to the other constraints, these are very easy to include! The only elements you will need to add to the navigation system's code are an initial location weighting, and a final location weighting. As done previously, the solver can be rewarded by finding routes which start and end on Mars. 


## "Start and end location" constraint 
All that has to be done in this unit is assing negative costs for starting and ending in a specific location. Departure from the home base on Mars is represented by $x_0=1$. Returning to home base after $N$ trips is given by x_{N^2}=1. Assigning these specific variables negative weights in the cost function will promote the solver to give them that value. 

This weighting procedure can also used to integrate user knowledge into the cost function. For example, visiting Venus after the $5$-th trip can be accomplished by negatively weighting $x_{3+5N}$. Alternatively, the weighting procedure can be used to promote a particular set of locations after a trip! For example, if you want to include a constraint about where the spaceship can and can't mine or sell after a trip, then you can negatively weight those set of locations. 

In the second unit it was stated that we want to keep the cost function "balanced", avoiding re-weighting by reverse combinations and giving all constraint terms equal weights (currently all have 0 weights). The benefit is that it is easier to specify preferred locations here, without having to tune each variable individually, as their relative weightings to the cost function are all the same. This will also reduce the effort in tuning in upcoming units. 

>[Note!]
>The negative weights assigned for these variables here will contribute to the negative weights assigned to the respective variables in the `no dissapearing constraint`. As a best-practice, you can combine these terms to reduce the length of the cost function. 

Another way of enforcing the `start and end location constraint` is to directly tell the solver to assign the respective variables a value equal to 1. The [`Problem.set_fixed_variables`](/azure/quantum/optimization-problem) is handy when you know what values certain variables **have** to be. Also, the cost function is automatically simplified for you when calling the method as the variables are filled in. The reason why the method is not implemented in this module is because you would not be able to encourage the spaceship to choose between a set of locations for a trip, a consequence of hard-coding the variable values. Even though promoting a set of locations is not part of the problem statement, it is an easy expansion of the `start and end location constraint` considered in this unit. 


### Progress on the navigation system's cost function

- The cost function contains:
  - The `travel costs`.
  - The `one location at a time constraint`, with constraint weight $w_1$. 
  - The `no dissapearing constraint`, with constraint weight $w_2$.
  - The `no revisiting locations constraint`, with constraint weight $w_3$.
  - The `start and end locations constraints`, with constraint weights $w_4$ and $w_5$. 

$$ \text{Final cost function: } \underset{x_0, x_1,\dots,x_{(N^2+2N)}}{min} \left(\sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right)  +   w_1 \left( \sum_{l=0}^{N} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} x_{(i+Nl)} \cdot x_{(j+Nl)} \text{ with } \{ i,j | i<j \} \right) +w_2\left( \sum_{k=0}^{N(N+1)-1} x_k \right) + w_3 \left(  \large{\sum}_{p=0}^{N^2+N-1}\hspace{0.25cm} \large{\sum}_{f=p+N,\hspace{0.2cm} \text{stepsize: } N}^{N^2-1} \hspace{0.35cm} (x_p \cdot x_f) \right) + w_4 ( x_0)  + w_5 ( x_{N^2}) \right)  $$

## Coding the initial conditions

Here, we add negative weights to the start and end location variables. A route in which these locations are the start and finish location should have the lowest associated costs. 

``` python

    #############################################################################################                        
    ##### Begin at x_0 (Mars - home base)
    terms.append(
        Term(
            c = -int(0),
            indices = [0]   
        )
    )

    ############################################################################################                        
    ##### End at x_{N^2} (Mars - home base)
    terms.append(
        Term(
            c = -int(0),
            indices = [len(CostMatrix)*(len(CostMatrix))]   
        )    
    )


    return Problem(name="Spaceship navigation system", problem_type=ProblemType.pubo, terms=terms)

OptimizationProblem = OptProblem(CostMatrix)  


``` 


## Next steps

Well done! You have finished programming the constraints for navigation system. It should not be long before the spaceship crew can use it to find their way through the solar system. In the next units we will go over constraint weight tuning and submitting the minimization problem to the Azure solvers. 

