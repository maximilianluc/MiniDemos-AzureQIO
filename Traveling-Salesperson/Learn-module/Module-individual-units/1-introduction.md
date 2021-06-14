In the [Quantum Computing Foundations](/learn/paths/quantum-computing-fundamentals?azure-portal=true) learning path, you have helped the spaceship crew optimize its asteroid mining expeditions and reparations of critical onboard emergency systems. Now you have been asked to code the navigation system for the spaceship to optimize its travel routes through the solar system. The spaceship needs to visit numerous planets, moons, and asteroids to mine and sell the space rocks. In this module, you'll use the Azure Quantum service to minimize the travel distance of the spaceship. 

The design and implementation of the navigation system is a case of the [**traveling salesperson** problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem?azure-portal=true). The objective is to find a path through a network of nodes such that an associated cost, such as the travel time or distance, is as small as possible. You can find applications of the traveling salesperson problem, or slight modifications of it, in a variety of fields. For example, in logistics, chemical industries, control theory, and bioinformatics. Even in your daily life, when you want to know the order in which you should visit school, the cinema, the office, and the supermarket to go the shortest way.

In this module, we will cover the formulation of this **minimization problem** by modeling travel costs and penalty functions. Afterward, we will solve the problem using the Earth's Azure Quantum Optimization service. All the content will be explained in the context of a spaceship that has to travel through the solar system to mine and sell asteroids.


## Learning objectives

After completing this module, you will be able:

- to understand the traveling salesperson problem.
- to evaluate the problem complexity, solvers, and the tuning process. 
- to model travel costs of the traveling salesperson.
- to formulate problem constraints into a penalty functions.
- to represent the minimization problem using Azure Quantum.
- to use the Azure Quantum Optimization service to solve optimization problems.
- to read and analyze results returned by the Azure Quantum solvers. 


## Prerequisites

- The latest version of the [Python SDK for Azure Quantum](/azure/quantum/optimization-install-sdk?azure-portal=true)
- [Jupyter Notebook](https://jupyter.org/install.html?azure-portal=true)
- An Azure Quantum workspace
- Basic linear algebra, only needing vector-matrix multiplications.

If you don't have these tools yet, we recommend that you follow the [Get started with Azure Quantum](/learn/modules/get-started-azure-quantum/?azure-portal=true) module first.
