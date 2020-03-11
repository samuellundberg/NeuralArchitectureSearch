# samuel_nas

### This is my master thesis about evolutionary methods for Neural Architecture Search where I will search for Image Classifiers. The project can be divided into two parts. 

 1: Setting up a Search Space of hyperparameters and their posible values as an optimization problem.

 2: Finding the optimal hyperparameters for a given task. As the objective function often is the validation performance and you need to train a neural network to obtain the validation performance the objective function is considered hard to evaluate. And most often black-box optimizatio techniques are used to optimize it. I will make an Evoultionary Algorithm (my_hypermapper/scripts/evolution.py) to do so and then I will compare it with Bayesian Optimization (my_hypermapper/scripts/random_scalarizations.py) and Random Search.

As a code base I use my LTH supervisor's open source black-box optimization software HyperMapper https://github.com/luinardi/hypermapper. HyperMapper has Bayesian Optimization, Random Search and Multi-start Local Search solvers implemented. So my task is to define a NAS problem and construct a Evolutionary Algorithm that hopefully can compete with the existing solvers.

If you want to know more about Neural Architecture Search, check this out: https://arxiv.org/pdf/1808.05377.pdf
