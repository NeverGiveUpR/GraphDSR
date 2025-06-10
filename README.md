# GraphDSR
Official code of paper "Mathematical expression exploration with graph representation and generative graph neural network", published in Neural Networks, 2025.
Python version: Python 3.6.15  
System: Linux  

# Install the environment
Please install the required packages as follows:  
```
pip install -r requirements.txt
```

# Running  
The configurations can be modified in config.json document. 

```
python main.py --dataset Eq --expr nguyen1
```
Where '--dataset' sets the running dataset, "Eq" for public benchmarks, and "Fq" for AI Feynman datasets. "--expr" is the benchmark name; the default setting is "nguyen1". You can change the running benchmark by revising the dataset name and the expr name. For example, to run the I_6_2a equation in the AI Feynman dataset:
```
python main.py --dataset Fq --expr I_6_2a
```
The code will save the best searched graph in a directory "./new_results/jsonn{node_num}", it will automatically create the category if it does not exist, and the weights, node types, and adjacent matrix will be saved in json file.

In the saved json file, the learned expression and the fitting R^2 will also be saved. To recover the graph:
```
graph = GraphNet(None, None, operators)
graph.load_graph_from_json(path)
```
The learned best graph normally represents an expression with high complexity. To prune the graph net, run the following script:
```
python connection_optimize.py --dataset Eq --expr nguyen1 --num 3
```
Where "--num" indicates to prune the graph net num times.
