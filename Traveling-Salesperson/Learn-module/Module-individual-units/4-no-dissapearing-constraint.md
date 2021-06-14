Unfortunately, the navigation system so far is not really helpful. Due to the way we formulated the cost function, the spaceship is penalized for being 'anywhere' without regard to the mission objective of visiting each planet, moon and asteroid of the itinerary.
Recall that the cost function includes two functions so far, the cost for traveling, and the one location at a time constraint. Currently, the Azure solvers on Earth would consistently return location vectors containing only 0 elements because that would lead to no incurred travel costs and no constraint violation, even though the spaceship is 'nowhere'. 

> By looking at the code and previous sections you come to the conclusion that the minimal cost is obtained by setting all $x_v$ to 0.

It is your task to make sure that the spaceship does not dissapear in the navigation system. In this unit, we will cover the *"no dissapearing"* constraint. By incorporating negative penalty terms (rewards) in the cost function, the solver will be encouraged to return routes in which the spaceship will consistently be in some location in our solar system. These negative cost terms will locally decrease cost function value(s) for suitable solution configurations. The aim is that these function valleys in the optimization landscape become low enough for the solver to settle in, depending on the constraints and weighting. 


## "No dissapearing" constraint

By incorporating negative terms in the cost function, we can encourage the solver to return a particular set of solutions. To demonstrate this point, take the following optimization problem: 

$$f(\boldsymbol{x}) := \underset{x_0, x_1, x_2}{min} x_0 + x_1 - x_2,$$ 
$$\text{with } x_0, x_1, x_2 \in \{0,1\}.$$

The minimum value for this example is achieved for the solution $x_0$ = 0, $x_1$ = 0, $x_2$ = 1, with the optimal function value equal to -1. Here the negatively weighted term encourages $x_2$ to take a value 1 instead of 0, unlike $x_0$ and $x_1$. With this idea in mind, you can prevent the spaceship from dissapearing in the navigation system. 

If the spaceship has to visit $N+1$ locations (+1 because return to home base) for a mining expedition, then $N+1$ of the $x_v$ should be assigned a value 1. Written as an equation for the $N(N+1)$ variables of a route ($N+1$ vectors, each with with $N$ elements gives $N(N+1)$):

$$ \sum_{v=0}^{N(N+1)-1} x_v = N+1.$$

You could split this equation for each location vector separately, but if you keep the constraint linear in $x_k$ then the resulting cost function will be the same. Moving the variables to the right side of the equation for negative weighting gives:

$$ 0 = (N+1) -\left( \sum_{v=0}^{N(N+1)-1} x_v \right).$$

As you may be aware, there is no guarantee which particular $x_v$ the solver will assign a value 1 in this equation. However, in the previous constraint the spaceship is already forced to be in a maximum of one location at a time. Therefore, with the cost function weights tuned, it can be assumed that only one $x_v$ per location vector will be assigned a value 1. 

To summarize, the `one location at a time constraint` penalizes the solver for being in more than one node at a time, while the `no dissapearing constraint` rewards the solver for being at as many locations as possible. The weights of the constraints will effectively determine how they are satisfied, a balance between the two needs to be found such that both are adhered to.

>[!Note]
> You can combine the `one location at a time constraint` and `no dissapearing constraint`. For explanatory purposes these two are split, however if you realize that -$x_v^2$ = -$x_v$, then you can expand the 'if' statement in the previous constraint for 'i==j'. 

### Progress on the navigation system's cost function

- The cost function contains:
  - The `travel costs`.
  - The `one location at a time constraint`, with constraint weight $w_1$. 
  - The `no dissapearing constraint`, with constraint weight $w_2$.

$$ \text{Cost function so far: } \underset{x_0, x_1,\dots,x_{(N^2+2N)}}{min} \left(\sum_{k=0}^{N-1}\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \left( x_{Nk+i}\cdot x_{N(k+1)+j}\cdot c_{i,j} \right)  +   w_1 \left( \sum_{l=0}^{N} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} x_{(i+Nl)} \cdot x_{(j+Nl)} \text{ with } \{ i,j | i<j \} \right) +\left( \sum_{v=0}^{N(N+1)-1} x_v \right) \right).$$


## Coding the constraint

Incorporating this constraint in the spaceship's navigation system's code is straightforward. We simply need to weight each variable by a negative weight. The $N+1$ can be ignored for the cost function, as it is independent of the variables optimized for and because it has no effect on the returned results. This scalar term is a constant linear offset in the cost for all solution configurations, and you can visualize it as a upward/downward shift of the entire optimization landscape.

``` python

    ############################################################################################
    ##### Constraint: No dissapearing constraint - encourage the spaceship to be 'somewhere' otherwise all x_k might be 0 (for example).
    for v in range(0, len(CostMatrix) + len(CostMatrix) * (len(CostMatrix))):    # Select variable 
        terms.append(
            Term(
                c = -int(0),          
                indices = [v]   
            )
        )
        ##----- Uncomment one of the below statements if you want to see how the weights are assigned! -------------------------------------------------------------------------------------------------
        #print(v)
        #print(f'No dissapearing constraint 2: x_{v} assigned weight: 0')                                                      # In a format for the solver (as formulated in the cost function)
        #print(f'No dissapearing constraint 2: location_{v % NumLocations} after {np.floor(v / NumLocations)} trips assigned weight: 0')   # In a format that is easier to read for a human

    return Problem(name="Spaceship navigation system", problem_type=ProblemType.pubo, terms=terms)

OptimizationProblem = OptProblem(CostMatrix)  


``` 

## Next steps

By now, the navigation system should prevent the spaceship from being in multiple locations at once, prevent it from dissapearing, and select a low-cost travel route. Great work! Yet, for the mining mission it is crucial that no planets, moons, or asteroids are not visited more than once. That would be bad for business! As you can see from the travel cost matrix, it is also the case that staying in the same planet has a zero-weighted cost (the diagonal terms of the matrix). The time has come for you to start looking at route specific constraints. 




