## Parallel Predictive Entropy Search for Multi-objective Bayesian Optimization with Constraints Spearmint branch.
========================================================================================

Spearmint is a Python 2.7 software package to perform Bayesian optimization. The Software is designed to automatically run experiments (thus the code name spearmint) in a manner that iteratively adjusts a number of parameters so as to minimize some objective in as few runs as possible.

## IMPORTANT: Please read about the license
Spearmint is under an **Academic and Non-Commercial Research Use License**.  Before using spearmint please be aware of the [license](LICENSE.md).  If you do not qualify to use spearmint you can ask to obtain a license as detailed in the [license](LICENSE.md) or you can use the older open source code version (which is somewhat outdated) at https://github.com/JasperSnoek/spearmint.  

## IMPORTANT: You are off the main branch! This branch contains the PPESMOC BO method.
This is the PPESMOC branch. This branch contains the Parallel Predictive Entropy Search for Multiobjective Optimization with Constraints. A PES extension that deals with the parallel constrained multi-objective scenario. If you have any doubts about the kind of problems that solves PPESMOC, please read the article that is listed above or send an email to eduardo.garrido@uam.es, main developer of this branch:

PPESMOC selects, at each iteration, a batch of input locations at which to evaluate the black-boxes, in parallel, to maximally reduce the entropy of the
problem’s solution. The best feature of PPESMOC is that the acquisition function is optimized through gradients obtained by the Autograd tool, enabling it to support batches of a high number of points (B=2,4,8,20,50...). In particular, PPESMOC computes the acquisition function faster than greedy parallel methods for a high number of batch points, a feature that can be useful for scenarios where a high number of nodes in a cluster are available to solve the optimization problem!

####Relevant Publications

Spearmint implements a combination of the algorithms detailed in the following publications:

    Practical Bayesian Optimization of Machine Learning Algorithms  
    Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams  
    Advances in Neural Information Processing Systems, 2012  

    Multi-Task Bayesian Optimization  
    Kevin Swersky, Jasper Snoek and Ryan Prescott Adams  
    Advances in Neural Information Processing Systems, 2013  

    Input Warping for Bayesian Optimization of Non-stationary Functions  
    Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams  
    International Conference on Machine Learning, 2014  

    Bayesian Optimization and Semiparametric Models with Applications to Assistive Technology  
    Jasper Snoek, PhD Thesis, University of Toronto, 2013  
  
    Bayesian Optimization with Unknown Constraints
    Michael Gelbart, Jasper Snoek and Ryan Prescott Adams
    Uncertainty in Artificial Intelligence, 2014

    Predictive Entropy Search for Multi-objective Bayesian Optimizaton
    Daniel Hernandez-Lobato, Jose Miguel Hernandez-Lobato, Amar Shah and Ryan Prescott Adams
    NIPS workshop on Bayesian optimization, 2015

This branch specifically includes the method that is described in the following preprint (soon to be updated on arxiv!):

    Parallel Predictive Entropy Search for Multi-objective Bayesian Optimization with Constraints
    Eduardo C. Garrido-Merchan, Daniel Hernandez-Lobato
    arXiv.


Follow the next steps to install Spearmint and execute a toy PPESMOC example:

### STEP 1: Installation and dependencies.
1. Download/clone the spearmint code
2. Install the spearmint package using pip: "pip install -e \</path/to/spearmint/root\>" (the -e means changes will be reflected automatically)
3. Download and install MongoDB: https://www.mongodb.org/
4. Install the pymongo package using e.g., pip or anaconda
5. Install PyGMO package (this is used for solving inner multi-objective optimization problems with known, simple and fast objectives). Follow the instructions of https://esa.github.io/pygmo/install.html
6. Try to compile Spearmint going into the main folder and execute "sudo python setup.py install". If the installation fails, that means that you lack an additional dependence of Spearmint like numpy or matplotlib. Install it using pip. Try to install spearmint again.

### STEP 2: Setting up your experiment
1. Create a callable objective function. See ../examples/moo/branin.py as an example.
2. Create a config file. See ../examples/moo/config.json as an example. Here you will see that we specify the PESM acquisition function. Other alternatives are ParEGO, EHI, SMSego and SUR.

### STEP 3: Running spearmint
1. Start up a MongoDB daemon instance: mongod --fork --logpath \<path/to/logfile\> --dbpath \<path/to/dbfolder\>
2. Run spearmint: "python main.py \</path/to/experiment/directory\>"
(Try >>python main.py ../examples/toy)

### STEP 4: Looking at your results
Spearmint will output results to standard out / standard err and will also create output files in the experiment directory for each experiment. In addition, you can look at the results in the following ways:

1. The results are stored in the database. The program ../examples/moo/generate_hypervolumes.py extracts them from the database and computes some
perforamnce metrics, e.g., using the hypervolume.

### STEP 5: Cleanup
If you want to delete all data associated with an experiment (output files, plots, database entries), run "python cleanup.py \</path/to/experiment/directory\>"

#### (optional) Running multiple experiments at once
You can start multiple experiments at once using "python run_experiments.py \</path/to/experiment/directory\> N" where N is the number of experiments to run. You can clean them up at once with "python cleanup_experiments.py \</path/to/experiment/directory\> N". 
