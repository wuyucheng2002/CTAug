# **Cohesive Topology Augmentation for Graph Contrastive Learning**

## CTAug Framework



## Requirements

- python  3.8
- torch  1.12.0
- torch-geometric  2.0.4
- PyGCL  0.1.2
- networkx  2.8.4

## File directory

```
.
├─ run.py
│
├─ CTAug
│  ├─ evaluate.py
│  ├─ model.py
│  ├─ utils.py
│  ├─ __init__.py
│  │
│  ├─ CTAug_GCA
│  │  ├─ CTAug_GCA_R.py
│  │  ├─ GCA.py
│  │  ├─ __init__.py
│  │  ├─ param
│  │  ├─ pGRACE
│  │  └─ simple_param
│  │
│  └─ CTAug_GCLs
│     ├─ CTAug_GraphCL_R.py
│     ├─ CTAug_GraphCL_S.py
│     ├─ CTAug_MVGRL.py
│     ├─ GraphCL.py
│     ├─ InfoGraph.py
│     ├─ JOAO.py
│     ├─ JOAOv2.py
│     ├─ MVGRL.py
│     └─ __init__.py
│
├─ data
│  ├─ Amazon-Computers
│  ├─ Coauthor-CS
│  ├─ Coauthor-Phy
│  ├─ IMDB-BINARY
│  ├─ IMDB-MULTI
│  └─ REDDIT-BINARY
│
└─ log
   └─ CTAug_GraphCL_R_42
      └─ IMDB-BINARY_kcore_0.25.log
```

- **run.py** is the running interface, we can use different methods and settings by passing different arguments.
- **CTAug** package implements all graph-level and node-level graph representation learning methods adopted in our experiments.
  - **CTAug_GCA** includes 1) node classification baseline method: *GCA* borrowed from https://github.com/CRIPAC-DIG/GCA, and 2) corresponding cohesive topology augmentation GCL method: *CTAug_GCA_R*.
  - **CTAug_GCLs** includes 1) graph classification baseline methods: _GraphCL_, _InfoGraph_, _JOAO_, and *JOAOv2* implemented by PyGCL(https://github.com/PyGCL/PyGCL); *MVGRL* borrowed from https://github.com/kavehhassani/mvgrl. 2) Cohesive topology augmentation GCL methods: *CTAug_GraphCL_R*, *CTAug_GraphCL_S*, *CTAug_MVGRL*.
  - **evaluate.py, model.py, and utils.py** store a series of functions and classes used for evaluating graph embedding, building contrastive learning model, preprocessing datasets, and extracting cohesive subgraph.

- **data** folder stores raw and processed data files, including 3 graph classification datasets (*IMDB-BINARY*, *IMDB-MULTI*, *REDDIT-BINARY*) and 3 node classification datasets (*Coauthor-CS*, *Coauthor-Physics* and *Amazon-Computers*), which are used in our experiments.
- **log** folder records experiment output, e.g., loss of each epoch, and mean and standard deviation of accuracies. When running a certain python file, it will create a subfolder in the log folder automatically, and the name of log file contains the name of the dataset we use and other setting information.
  - 'CTAug_GraphCL_R_42/IMDB-BINARY_kcore_0.25.log' is an output example. It is a log record for *CTAug_GraphCL_R* method, which set random seed at 42, used *IMDB-BINARY* dataset, chose k-core property, and set probability decay factor at 0.25.

## Arguments

| Name       | Default value   | Description                                                                                                                                                                |
|:-----------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| seed       | 42              | Random seed.                                                                                                                                                               |
| epoch      | 100             | Training epoch.                                                                                                                                                            |
| interval   | 20              | Interval epoch to test.                                                                                                                                                    |
| batch_size | 64              | Batch size of dataset partition.                                                                                                                                           |
| shuffle    | True            | Shuffle the graphs in the dataset or not.                                                                                                                                  |
| save       | False           | Whether to save the model.                                                                                                                                                 |
| eva        | True            | Evaluate immediately or save the model.                                                                                                                                    |
| times      | 5               | The number of repetitions of the experiment.                                                                                                                               |
| save_path  | None            | The name of the folder to save log.                                                                                                                                        |
| device     | None            | Running environment, can be cpu or cuda.                                                                                                                                   |
| hid_units  | 64              | Dimension of hidden layers and embedding.                                                                                                                                  |
| early_stop | False           | Whether early stop or not.                                                                                                                                                 |
| patience   | 3               | Patience for early stop.                                                                                                                                                   |
| pn         | 0.2             | The probability of dropping node, removing edge, or sampling subgraph.                                                                                                     |
| aug_name   | ND              | Augmentation ways, can be chosen from {ND, ER, SUB}, connected with '+', e.g., 'ND+ER'. (ND: node dropping, ER: edge removal, SUB: random-walk-based sampling.)            |
| graph_num  | 20              | The number of pre-set candidate augmented graph.                                                                                                                           |
| core       | kcore           | Subgraph property, can be chosen from {kcore, ktruss, random}, random means randomly choose from kcore and ktruss.                                                         |
| kcore      | 1.0             | Pre-set weight of kcore property.                                                                                                                                          |
| ktruss     | 0               | Pre-set weight of ktruss property.                                                                                                                                         |
| random     | 0               | Pre-set weight of random (not use property).                                                                                                                               |
| sim        | Jaccard         | Choose the similarity metric from {Euclidean, Cosine, Jaccard}.                                                                                                            |
| cal_weight | node            | Choose the edge weight calculation strategy from {node, edge}.                                                                                                             |
| frac       | 0.25            | The decay factor of dropping probability in CTAug R-mode.                                                                                                                  |
| method     | CTAug_GraphCL_R | Use different contrastive learning methods. Can be chosen from {GraphCL, MVGRL, JOAO, JOAOv2, InfoGraph, CTAug_GraphCL_S, CTAug_GraphCL_R, CTAug_MVGRL, GCA, CTAug_GCA_R}. |
| dataset    | IMDB-BINARY     | Dataset name, can be chosen from graph classification: {IMDB-BINARY, IMDB-MULTI, REDDIT- BINARY}, node classification: {Coauthor-CS, Coauthor-Phy, Amazon-Computers}.      |
| param      | default         | Hyper-parameters for GCA/CTAug_GCA_R, can be chosen from {local:coauthor_cs.json, local:coauthor_phy.json, local:amazon_computers.json}.                                   |

We can choose different methods by passing value to the '**method**' arguments. For different methods, the corresponding effective and essential arguments are as follows:

| Arguments  | GraphCL | JOAO | JOAOv2 | InfoGraph | CTAug_GraphCL_S | CTAug_GraphCL_R | MVGRL | CTAug_MVGRL | GCA  | CTAug_GCA_R |
| :--------- | :-----: | :--: | :----: | :-------: | :-------------: | :-------------: | :---: | :---------: | :--: | :---------: |
| seed       |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |  √   |      √      |
| epoch      |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| interval   |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| batch_size |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| shuffle    |    √    |  √   |   √    |     √     |        √        |        √        |       |             |      |             |
| save       |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| eva        |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| times      |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| save_path  |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| device     |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |  √   |      √      |
| hid_units  |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| early_stop |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| patience   |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |      |             |
| pn         |    √    |  √   |   √    |           |        √        |        √        |       |             |      |             |
| aug_name   |    √    |      |        |           |        √        |                 |       |             |      |             |
| graph_num  |         |      |        |           |        √        |                 |       |             |      |             |
| core       |         |      |        |           |        √        |        √        |       |             |      |      √      |
| kcore      |         |      |        |           |                 |                 |       |      √      |      |             |
| ktruss     |         |      |        |           |                 |                 |       |      √      |      |             |
| random     |         |      |        |           |                 |                 |       |      √      |      |             |
| sim        |         |      |        |           |        √        |                 |       |             |      |             |
| cal_weight |         |      |        |           |                 |                 |       |             |      |             |
| frac       |         |      |        |           |                 |        √        |       |             |      |      √      |
| method     |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |  √   |      √      |
| dataset    |    √    |  √   |   √    |     √     |        √        |        √        |   √   |      √      |  √   |      √      |
| param      |         |      |        |           |                 |                 |       |             |  √   |      √      |


## Running examples

- **Example 1: Run *CTAug-GraphCL-S* on *IMDB-BINARY* dataset (use k-core property).**

  ```shell
  python run.py --method=CTAug_GraphCL_S --dataset=IMDB-BINARY --core=kcore
  ```
  
  - The output path is 'log/CTAug_GraphCL_S_42/IMDB-BINARY_kcore_Jaccard_20.log', means we use *CTAug_GraphCL_S* method, set random seed at 42, choose *IMDB-BINARY* dataset, use k-core property, set Jaccard similarity as similarity metric, and the pre-set augmentation number is 20.
  - The log file will record the loss every epoch and the graph properties (including the loose factor) we choose for every batch, calculate the prediction accuracy per 20 epoch (the total training epoch is 100), and get the mean and standard deviation for repeated experiments.
  
- **Example 2: run *CTAug-GCA-R* on *Coauthor-CS* dataset (use k-truss property, set decay factor to 0.5).**

  ```shell
  python run.py --method=CTAug_GCA_R --dataset=Coauthor-CS --param=local:coauthor_cs.json --core=ktruss --frac=0.5
  ```
  
  - The output path is 'log/CTAug_GCA_R_42/Coauthor-CS_ktruss_0.5.log', means we use *CTAug_GCA_R* method, set random seed at 42, choose *Coauthor-CS* dataset, use k-truss property, and set decay factor to 0.5.
  - The log file will record the loss and the graph properties (including the loose factor) we choose every epoch, calculate the prediction accuracy per 50 epoch, and get the mean and standard deviation for repeated experiments. 
  - The hyper-parameter setting is the same as original [GCA](https://github.com/CRIPAC-DIG/GCA) code by using '--param=local:coauthor_cs.json'.
