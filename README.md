# Keywords_extraction_with_GOW
Graph of words and keywords extraction 

***Graph of words and keywords extraction.ipynb***

This notebook is dedicated to people who want to **extract keywords** from **text document** or corpus documents using a **graph approach**.

The goal of this notebook is to extract keywords from a text file using three different approachs :
- Best Coverage keywords extraction - http://www2013.w3c.br/proceedings/p715.pdf
- Div Rank keywords extraction - http://clair.si.umich.edu/~radev/papers/SIGKDD2010.pdf
- K-core Number 


***K-core on tokennized data***

This python code takes as input a input_root containing directories of "files.txt" and subdirectories.
The "files.txt" are tokenized data in the following form : (word,stemmed_word,pos_tag)

We visit all this files and perform a k-core approach for each file. 
We recreate the same files and folders tree and save the .csv of keywords and graph for density and inflexion method.

The parameters of this python code are :

- arg1 : input_root (from where we extract k_core graphs and keywords)

- arg2 : output_root (where to save it)

- arg3 : - 1 if all the files has to be treated independantly - 0 if we want one graph and one keywords .csv file in output (analyzed with all the files)
- arg4: - 1 for words in lower case - 0 otherwise
            
- arg5: (int): the value of the windows size. A window size of 5 means that an edge will be created between all the words separated from less than 5 words in the text

- arg6...n: the pos tag to keep (see the script) (ADJ,NOUN,VERB...)
