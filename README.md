## Parallel Predictive Entropy Search for Multi-objective Bayesian Optimization with Constraints Spearmint branch.
========================================================================================

Spearmint is a Python 2.7 software package to perform Bayesian optimization, a class of methods that deliver state of the art results when performing the hyper-parameter tuning of machine learning algorithms. The Software is designed to automatically run experiments (thus the code name spearmint) in a manner that iteratively adjusts a number of parameters defined in an input space so as to minimize some objective (or a set of objectives) in as few runs as possible, assuming that each evaluation is costly, noisy and we lack its analytical expression.

## IMPORTANT: Please read about the license
Spearmint is under an **Academic and Non-Commercial Research Use License**.  Before using spearmint please be aware of the [license](LICENSE.md).  If you do not qualify to use spearmint you can ask to obtain a license as detailed in the [license](LICENSE.md) or you can use the older open source code version (which is somewhat outdated) at https://github.com/JasperSnoek/spearmint.  

## IMPORTANT: You are off the main branch! This branch contains the PPESMOC BO method.
This is the PPESMOC branch. This branch contains the Parallel Predictive Entropy Search for Multiobjective Optimization with Constraints. A PES extension that deals with the parallel constrained multi-objective scenario. If you have any doubts about the kind of problems that solves PPESMOC, please read the article that is listed above or send an email to eduardo.garrido@uam.es, main developer of this branch. Caution: The branch is a stable version but needs to be cleaned and refined. Will be updated soon with these changes.

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
6. Try to compile Spearmint going into the main folder and execute "sudo python2 setup.py install". If the installation fails, that means that you lack an additional dependence of Spearmint like numpy or matplotlib. Install it using pip. Try to install spearmint again.

### STEP 2: Setting up your parallel constrained multi-objective experiment and solving it via the PPESMOC acquisition function.
0. This step assumes that your problem belongs to the parallel constrained multi-objective setting. To see the configuration of other scenarios, like the vanilla one that optimizes a single objective funciton, see other examples of the ../examples folder.
1. Go to the ../examples/mocotoy_ppesmoc/ folder to see an example.  
2. Configure your problem: Create a config.json file as the ../examples/mocotoy_ppesmoc/config.json file. Specify the input space changing the variables entry. GPs are guaranteed to work for problems with less than 8 dimensions. Specify the number of BO iterations changing the max_finished_jobs entry and the batch size via the batch_size entry. You can also specify the number of GP samples via the mcmc_iters entry (default 10). Finally, specify the objectives and constraints via the tasks entry as in the ../examples/mocotoy_ppesmoc/config.json file.
3. Integrate your black-box in Spearmint! Create a xxxx.py file where xxxx is the main_file entry of the config.json file. Spearmint will call the def main(job_id, params) method of that file in a sequential loop. Params is a dictionary that will hold the Spearmint recommended params of each iteration. The keys of the params are the ones specified in the variables entry of the config.json. Now make your cool computations there (like evaluating your machine learning algorithm hyper-parametrized by the hyper-parameters that you retrieve via the params dictionary or other objectives such as the prediction time and some constraints like the size of the file where you save your model). In order to send the values of your objectives and constraints back to Spearmint, return the values in a dictionary where the keys specify each of the black-boxes.

### STEP 3: Running spearmint
1. Start up a MongoDB daemon instance: mongod --fork --logpath \<path/to/logfile\> --dbpath \<path/to/dbfolder\> or alternatively start a MongoDB daemon every time that you start your machine. Spearmint will store there the results of each iteration of the experiment.
2. Run Spearmint and specify the experiment that it needs to solve: "python main.py \</path/to/experiment/directory\>"
(Try >>python main.py ../examples/mocotoy_ppesmoc) Alternatively, you can be in the experiment folder and run Spearmint from there (python ../../spearmint/main.py .)
3. Wait until Spearmint finishes the evaluations. Depending on your setting, that can be a costly process! (It will print a have a nice day trace and the end of the experiment). The input space configurations (feasible Pareto sets that correspond to, for example, hyper-parameters or other configurations) recommended by Spearmint will be stored on MongoDB.

### STEP 4: Looking at the results of your experiment and evaluating them.
Spearmint will output results to standard out / standard err and will also create output files in the experiment directory for each experiment. In addition, you can look at the results in the following ways:

1. The recommendations are stored in the database. The program ../examples/mocotoy_ppesmoc/generate_hypervolumes.py extracts them from the database and computes some perforamnce metrics, e.g., using the hypervolumes of them. Run that file and wait until the hypervolumes are computed, check the hypervolumes.txt file and check which has been the line with more hypervolume. Check the output/ folder for the file with the number of that iteration and that will be the hyper-parameter values that optimize your constrained multi-objective problem!

### STEP 5: Cleanup
If you want to delete all data associated with an experiment (output files, plots, database entries), run "python cleanup.py \</path/to/experiment/directory\>" 
