# Keywords_extraction_with_GOW

- First we present an example of the methods used to extract keywords (see **Graph of words and keywords extraction.ipynb** and **K-truss_code_example.ipynb**)
- Then we give a code to compute the k_core and obtain the graphs of directories of files or all files in directories containing sub-directories (see **K_core_corpus.py**)
- We also give an implementation of the K-truss algorithm (see **K-truss_code.py**)
- We make a time analysis to see the evolution of some words through time, in order to detect events related to them. 

## Libraries 

- Networkx to create and vizualize graphs 
- Spacy to preprocess the text 

## Papers implemented : 
- The k-core is directly taken from Networkx library.
- The k-truss is implemented following https://arxiv.org/abs/1205.6693
- The density and inflexion methods are implemented following https://www.aclweb.org/anthology/D16-1191

## ***Graph of words and keywords extraction.ipynb***

This notebook is dedicated to people who want to **extract keywords** from **text document** or **corpus documents** using a **graph approach**.

The goal of this notebook is to extract keywords from a text file using four different approachs :
- Best Coverage keywords extraction - http://www2013.w3c.br/proceedings/p715.pdf
- Div Rank keywords extraction - http://clair.si.umich.edu/~radev/papers/SIGKDD2010.pdf
- K-core Number - Python Library Networkx
- K-Truss - https://arxiv.org/abs/1205.6693

Through a french summary of Games of Thrones, we bring an example of the outputs of the four different approaches.


## ***K-core corpus***

This python code takes as input a input_root containing directories of "files.txt" and subdirectories.
The "files.txt" are tokenized data in the following form : (word,stemmed_word,pos_tag)

We visit all this files and perform a k-core approach for each file. 
We recreate the same files and folders tree and save the .csv of keywords and graph for both density and inflexion methods.

The parameters of this python code are :

- arg1 : input_root (from where we extract k_core graphs and keywords)

- arg2 : output_root (where to save it)

- arg3 : - 1 if all the files has to be treated independantly - 0 if we want one graph and one keywords .csv file in output (analyzed with all the files)
- arg4: - 1 for words in lower case - 0 otherwise
            
- arg5: (int): the value of the windows size. A window size of 5 means that an edge will be created between all the words separated from less than 5 words in the text

- arg6...n: the pos tag to keep (see the script) (ADJ,NOUN,VERB...)

This script contains the **Inflexion method function** and the **Density method function**


## K-truss_code_example.ipynb

This jupyter notebook is an example of the following script 

## K-truss code 

Two functions are implemented. 

- The first one compute the K-truss of each node in G, the maximum non empty subgraph, the k from the maximum non empty subgraph and the necessary informations to compute the density and inflexion method. 
- The second one gives the k-truss subgraph of the graph, where k is given as an input


