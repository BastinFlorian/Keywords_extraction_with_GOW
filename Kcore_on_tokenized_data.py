###### EXAMPLES ######

# To try this code on your terminal :

## EX :  python3 Keywords_text.py /input/dir  /output/dir  (1 if graph for every file 0 if graph for all files) (1 if lower case)  window_size  pos_tag_to_keep1 .. pos_tag_to_keep_n

### Ex 1:

## python3 Keywords_text.py /home/my_dir_with_subdir /home/output/same_arborescence_created_with_results 1 1 10 NOUN VERB ADJ 

#Will produce for each file contained in /home/my_dir_with_subdir and in its sub_directories two csv with the kcore keywords of dens and inf values and their png graph
#If there is 10 files in the sub-dirs of the input files 20 csv created with the format (words,kcore_values_in_reversed_order,type_method(dens or inf)).
#The words will be in lower case, only the NOUN,VERB,ADJ will be kept and the windows size for kcore will be 10.

### Ex 2:

## python3 Keywords_text.py /home/my_dir_with_subdir /home/output/same_arborescence_created_with_results 0 1 8  NOUN VERB 

# Will produce two csv for the dens and inf methods and two graph only. A graph represents all the files in the dirs and sub_dirs on the input_file
# Words in lower case and only Noun and Verb are kept

### Pos_tag_to_keep can be the following : 

#ADJ: adjective
#ADP: adposition
#ADV: adverb
#AUX: auxiliary
#CCONJ: coordinating conjunction
#DET: determiner
#INTJ: interjection
#NOUN: noun
#NUM: numeral
#PART: particle
#PRON: pronoun
#PROPN: proper noun
#PUNCT: punctuation
#SCONJ: subordinating conjunction
#SYM: symbol
#VERB: verb
#X: other
###

###### END EXAMPLES ######

#####################--CODE--#####################

import os 
import re 
import sys
import pandas as pd
import itertools
import networkx as nx
from networkx import core_number
from networkx import k_core
#import matplotlib.pyplot as plt
import numpy as np

### INPUT 
stop_words=['l','n','d','a','l','.','·','T.','·']
input_root=sys.argv[1]
root_output=sys.argv[2]
results_for_each_file=int(sys.argv[3])
lower=int(sys.argv[4])
w=int(sys.argv[5])
pos_tag_to_keep=b=[str(x) for x in sys.argv[6:]]
### END INPUT


######## LIST OF OF THE FILENAMES CONTAINED IN THE DIRECTORY AND SUB-DIRECTORIES OF "INPUT ROOT" ######## 

def get_filename_list(filename):
    if (os.path.isdir(filename)):
        print("get_list_filename_start")
        path=[]
        for root, dirs, files in os.walk(filename, topdown = False):
            for name in files:
                path.append(os.path.join(root,name))
    else:
        path=[filename]   
    print("get_list_filename_finished")
    return(path)
# RETURN A LIST OF FILENAMES TO STUDY

######## GET A A LIST OF LIST CONTAINING THE WORDS TO KEEP, ie,a word not in the STOPWORDS and  with its pos_tag in "POS_TAG_TO_KEEP" ########

#paths_files is a df with three columns ("words","stemm","pos_tag")
def get_full_text(path,pos_tag_to_keep,lower):
    full_text=[]
    for path_files in path:
        try:
            text=pd.read_csv(path_files,sep='\t', header=None)
            text=text.loc[text[2].isin(pos_tag_to_keep)]
            text=text.loc[~text[0].isin(stop_words)]
            text_lower=[]
            if(lower):
                for word in list(text[0]):
                    text_lower.append(word.lower())
                full_text.append(text_lower)
            full_text.append(list(text[0]))
        except:
            empty_list=open("/home/output/empty_list.txt","a")
            empty_list.write('{} \n'.format(path_files))
            empty_list.close()            
    print('full_text_finished')
    return(full_text)
#RETURN [[Text1_W1,...,Text1_Wn],...,[Text2W1,...,Text2Wn]]

######## CREATE THE GRAPH EDGES CONTAINED IN "from_to" in the following format = (word1,word6):weight,... ########

# if window stop at the end of a line stopping_end_of_line=1, else, equal to 0
# if results_for_each_file=0, stopping end of line =1
def terms_to_graph_sents(clean_txt_sents, w,stopping_end_of_line=0):
    print("term_to_graph_start")
    from_to = {}
    if(not stopping_end_of_line):
        extended_clean_txt_sents=[]
        for sublist in clean_txt_sents:
            extended_clean_txt_sents.extend(sublist)
        clean_txt_sents=[extended_clean_txt_sents]
    for k,sents in enumerate(clean_txt_sents):
        print(k/len(clean_txt_sents))
        len_sents=len(sents)

        # create initial complete graph (first w terms)
        terms_temp = sents[0:min(w,len_sents)]
        indexes = list(itertools.combinations(range(min(w,len_sents)), r=2))
        new_edges = []
        for my_tuple in indexes:
            new_edges.append(tuple([terms_temp[i] for i in my_tuple]))

        for new_edge in new_edges:
            if new_edge in from_to:
                from_to[new_edge] += 1
            else:
                from_to[new_edge] = 1
        if(w<=len_sents):
            # then iterate over the remaining terms
            for i in range(w, len_sents):
                considered_term = sents[i] # term to consider
                terms_temp = sents[(i-w+1):(i+1)] # all terms within sliding window
                # edges to try
                candidate_edges = []
                for p in range(w-1):
                    candidate_edges.append((terms_temp[p],considered_term))

                for try_edge in candidate_edges:

                    if try_edge[1] != try_edge[0]:
                    # if not self-edge
                        # if edge has already been seen, update its weight
                        if try_edge in from_to:
                            from_to[try_edge] += 1
                        # if edge has never been seen, create it and assign it a unit weight     
                        else:
                            from_to[try_edge] = 1
    print("term_to_graph_finished")
    return(from_to)
#EDGES_CREATED

####### NETWORKX_GRAPH ########

def weighted_graph(tuples_words_sents_weighted):
    print('G_start')
    G = nx.Graph()
    for keys,values in tuples_words_sents_weighted.items():
        G.add_edge(keys[0],keys[1],weight=values)
    print('G_finished')
    return(G)




###### ORDER RESULTS FOR THE OUTPUT CSV ######

def order_dict_best_keywords(G_core_number,nb_keys_terms_needed=-1):
    print("order_keywords_start")
    k_core_keyTerms=sorted(G_core_number, key=G_core_number.get, reverse=True)
    if(nb_keys_terms_needed==-1):
        Kcore_values=[G_core_number[x] for x in k_core_keyTerms]
    else:
        Kcore_values=[G_core_number[x] for x in k_core_keyTerms[:nb_keys_terms_needed+1]]
    print("order°keywords_finished")
    return(k_core_keyTerms,Kcore_values)

###### DENSITY METHOD --- see :http://www.lix.polytechnique.fr/Labo/Antoine.Tixier/EMNLP_2016_keyword.pdf (page 1864) ######

def dens_function(output_density):
    print("dens_start")
    D_n=[]
    for V_E in output_density:
        if(V_E[1]>1):
            D_n.append([V_E[0],V_E[2]/(V_E[1]*(V_E[1]-1))])
        else:
            D_n.append([V_E[0],0])
    print("dens_finished")
    return(D_n)

def elbow_function(D_n):
    print("Elbow start")
    if(len(D_n)>2):
        a_equation=(D_n[0][1]-D_n[-1][1])/(D_n[0][0]-D_n[-1][0])
        b_equation=(D_n[0][1]-D_n[0][0]*a_equation)
        distance={}
        s=0
        max_dist=0
        max_ind=0
        
        for (x,y) in D_n:
            distance[s]=(abs((a_equation*x+b_equation-y)/(((a_equation**2)+(1)))**(1/2)))
            if(distance[s]>max_dist):
                max_dist=distance[s]
                max_ind=x
            s+=1   
        distance=sorted(distance, key=distance.get, reverse=True)
        print("Elbow finished")
        if(distance[0]!=distance[1]):
            return(max_ind)
        else:
            return(0)
    else:
        if(len(D_n)==1):
            return(0)
        elif(D_n[0][0]==max(D_n[0][0],D_n[1][0])):
            return(0)
        else:
            return(1)

def output_dens(G):
    D_n=[]
    k=0
    GG=k_core(G, k, core_number=None)
    D_n.append([k,len(GG.nodes),len(GG.edges)])
    
    while(len(GG.nodes)>0):
        k+=1
        GG=k_core(G, k, core_number=None)
        D_n.append([k,len(GG.nodes),len(GG.edges)])
    return(D_n)
#END DENSITY METHOD

###### INFLEXION METHOD ######

def inflexion_method(output_density):
   CD=[]
   n=1
   for i in range(len(output_density)-1):
       CD.append(output_density[i+1][1]-output_density[i][1])
       for i in range(2,len(CD)-1)
           if(CD[i+1]<0 and CD[i]>0):
               n=i
   return(n)    
#END INFLEXION METHOD

###### GIVE OPTIMIZED NUMBER OF K_CORE TO STUDY ######

def get_optimized_nb_keywords(G,dens_methode,inf_method):   
    print("Optimized number ok keywors starts"
    if(dens_methode):

        output_density=output_dens(G)
        D_n=dens_function(output_density) 
        print("Optimized number of keywords finished")
        return(elbow_function(D_n))

    else:

        output_density=output_dens(G)
        n=inflexion_method(output_density)
        print("Optimized number of keywords finished")
        return(n)    



###### FUNCTION OF KEYWORDS EXTRACTION ###### 
def K_core_function(root,file_name,root_output,pos_tag_to_keep,lower,w):
    print("K_core start")

    if(file_name==None):
        filename=root
    else:
        filename=os.path.join(root,file_name)
        
    path=get_filename_list(filename)

    full_text=get_full_text(path,pos_tag_to_keep,lower)
    if(not len(full_text)==0):
        tuples_words_sents_weighted=terms_to_graph_sents(full_text, w=w,stopping_end_of_line=1)

        G_weighted=weighted_graph(tuples_words_sents_weighted)
        print("G_W_S")
        G_weighted.remove_edges_from(nx.selfloop_edges(G_weighted))
        print("G_W_F")

        inf_value=get_optimized_nb_keywords(G_weighted,dens_methode=False,inf_method=True)

        dens_value=get_optimized_nb_keywords(G_weighted,dens_methode=True,inf_method=False)
        print('Subgraph inf starts')
        inf_subgraph_k_core=k_core(G_weighted, inf_value, core_number=None)
        print('Subgraph inf finished')


        print('Subgraph dens starts')
        dens_subgraph_k_core=k_core(G_weighted, dens_value, core_number=None)
        print('Subgraph dens finished')

        print('Core nb inf starts')
        G_inf_nb=core_number(inf_subgraph_k_core)
        print('Core nb inf finished')

        print('Core nb dens starts')
        G_dens_nb=core_number(dens_subgraph_k_core)
        print('Core nb dens finished')

        print("Ordering...")
        dens_keyTerms,dens_values=order_dict_best_keywords(G_dens_nb)
        inf_keyTerms,inf_values=order_dict_best_keywords(G_inf_nb)

        print("Getting DataFrame")
        inf_df=pd.DataFrame(columns=['Keywords','Core_number','Inflexion_value'],data=np.array([inf_keyTerms,inf_values,[inf_value]*len(inf_keyTerms)]).T)
        dens_df=pd.DataFrame(columns=['Keywords','Core_number','Density_value'],data=np.array([dens_keyTerms,dens_values,[dens_value]*len(dens_keyTerms)]).T)

        if(len(inf_subgraph_k_core.nodes)>50):
            inf_subgraph_k_core=inf_subgraph_k_core.subgraph(inf_keyTerms[:50])

        if(len(dens_subgraph_k_core.nodes)>50):
            dens_subgraph_k_core=dens_subgraph_k_core.subgraph(inf_keyTerms[:50])

        if(file_name==None):
            name_doc_dens="{}/keywords_dens_w:{}_POS:{}.xls".format(root_output,w,'-'.join(pos_tag_to_keep))
            name_png_dens="{}/subgraph_dens_w:{}_POS:{}.png".format(root_output,w,'-'.join(pos_tag_to_keep))
            name_doc_inf="{}/keywords_inf_w:{}_POS:{}.xls".format(root_output,w,'-'.join(pos_tag_to_keep))
            name_png_inf="{}/subgraph_inf_w:{}_POS:{}.png".format(root_output,w,'-'.join(pos_tag_to_keep))
        
        else:
            if (not os.path.exists(os.path.join(root_output,'.'.join(file_name.split('.')[:-1])))):
                os.makedirs(os.path.join(root_output,'.'.join(file_name.split('.')[:-1])))
            name_doc_dens="{}/keywords_dens_w:{}_POS:{}.xls".format(os.path.join(root_output,'.'.join(file_name.split('.')[:-1])),w,'-'.join(pos_tag_to_keep))
            name_png_dens="{}/subgraph_dens_w:{}_POS:{}.png".format(os.path.join(root_output,'.'.join(file_name.split('.')[:-1])),w,'-'.join(pos_tag_to_keep))
            name_doc_inf="{}/keywords_inf_w:{}_POS:{}.xls".format(os.path.join(root_output,'.'.join(file_name.split('.')[:-1])),w,'-'.join(pos_tag_to_keep))
            name_png_inf="{}/subgraph_inf_w:{}_POS:{}.png".format(os.path.join(root_output,'.'.join(file_name.split('.')[:-1])),w,'-'.join(pos_tag_to_keep))
         print(name_doc_dens)
    if(len(dens_subgraph_k_core.nodes)>0):
        plt.figure()
        nx.draw(inf_subgraph_k_core, with_labels=True,font_color='k',node_color='g',edge_color='y',font_size=max(min(20,500/len(inf_subgraph_k_core.nodes)),9),width=1,node_size=0)
        plt.savefig(name_png_inf)
        plt.close('all')


        plt.figure()
        nx.draw(dens_subgraph_k_core, with_labels=True,font_color='k',node_color='g',edge_color='y',font_size=max(min(20,500/len(dens_subgraph_k_core.nodes)),9),width=1,node_size=0)
        plt.savefig(name_png_dens)
        plt.close('all')

    inf_df.to_csv(name_doc_inf)
    dens_df.to_csv(name_doc_dens)

###### MAIN ######

for root,dirs,files in os.walk(input_root):
    root_output_full=re.sub(r'^{0}'.format(re.escape(input_root)), root_output, root)
    for d in dirs: 
        root_output_dir=os.path.join(root_output_full,d)
        if not os.path.exists(root_output_dir):
            os.makedirs(root_output_dir)
    if(results_for_each_file): 
        for f in files:
            K_core_function(root,f,root_output_full,pos_tag_to_keep,lower,w)

if(not results_for_each_file):
    if(os.path.isdir(input_root)):
        root_output_full=root_output
        K_core_function(input_root,None,root_output_full,pos_tag_to_keep,lower,w)
    else:
        root_file='/'.join(input_root.split('/')[:-1])
        file_name=input_root.split('/')[-1]
        root_output_full=re.sub(r'^{0}'.format(re.escape(root_file)), root_output, root_file)
        K_core_function(input_root,file_name,root_output_full,pos_tag_to_keep,lower,w)

###### END MAIN ######
