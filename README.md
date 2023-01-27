# [Santa 2022 - The Christmas Card Conundrum](https://www.kaggle.com/competitions/santa-2022/overview)
### Optimize the configuration space for printing an image
See kaggle link for details, getting the data, etc.

The solution provided here achieved 55th place in the competition with a score of 74337. It visits each point once, except the origin.
### Install
If you want to play around with the code, you can pip install the repo with `python -m pip install -e .`

To use the LKH solver run `generate_lkh_graph.py` to get the graph file cca 4.5MB, the other files need for solver are already in the folder `tsp`.
### Initial approach
Inspired by [this notebook](https://www.kaggle.com/code/oxzplvifi/pixel-travel-map).

A greedy algorithm that visits an unvisited point every step by either picking:
* preferring one cardinal direction (i.e. up/down/left/right);
* the cheapest neighboring configuration; 
* a multistep path to the closest (L1 distance) unvisited point.

After visiting every point it returns the origin configuration. Preferring a cardinal direction turns out to be a great heuristic by reducing the number of times search gets stuck.

Also tried a variation of this where you start two searched simultaneously over the shared point space. After all points are visited, connect one search path to the other and reverse one to get the combined solution.

Grid search over all cardinal directions and various other hyperparameters never produced a score below 78000.

### TSP approach
Inspired by [this discussion](https://www.kaggle.com/competitions/santa-2022/discussion/376306).
##### Translation image to TPS graph and solving with LKH
Implemented a variation of the standard configuration described above. Translated entire image to a TSP graph. Edges were restricted to ensure moving from point to point could be expressed with a standard configuration. In general, each point was connected to all eight of its neighbors. There are some additional edges along the bounds, and very limited edges between different quadrants of the image.

Resulting graph and a naive initial solution were given to an [LHK solver](http://webhotel4.ruc.dk/~keld/research/LKH-3/)
and a solution was recovered.

##### Start and end optimization

The robotic arm starts from and returns to the origin configuration after painting the picture. Additionally, the origin configuration does not correspond to a standard configuration. To avoid visiting some pixels more than once, start and end custom path were found that followed the LKH solution, but converged to a standard configuration as quickly as possible.
