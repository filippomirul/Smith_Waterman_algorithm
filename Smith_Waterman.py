import numpy as np
import argparse
import textwrap


parser = argparse.ArgumentParser(
    formatter_class = argparse.RawDescriptionHelpFormatter,
    description = textwrap.dedent("""
    This code is an implementation of Smith-Waterman algorithm, it uses both matrix and a graph class.
    The matrix is constructed while the scores are being up dated (following the scores assignments), simultaneously for each possibility a node is created
    and the graph is custructed. The graph is needed for the later back track, so is oriented in the opposite way of the
    computations that creates the matrix.

    This algorithm return all the possible allignment with score equal to the maximum score.
    For this reason form the more are long the sequences in input the more time it needs to compute all the possibilities.

    For testing try :

    atcgatggac
    aggctatcaact

    """))

parser.add_argument("-s1", "--sequence_1", type = str, help = "First sequence")
parser.add_argument("-s2", "--sequence_2", type = str, help = "Second sequence")
parser.add_argument("-sc", "--scores", type = dict, help = """Dictionary with match, mis-match and gap scores.
                    Example: {"match": 3, "mis-match": -2, "gap" : 1} """) # TODO IMPLEMENT
args = parser.parse_args()


class Graph:

    """     
    This class create a graph with the use of a disctionary.
    The keys are the nodes id and the value is an other dictionary
    with node link with the fisrt as key and the weight as value.
    EX:
    {node1: {node2: 23, node 23: 45, node64: 23}, node2 : {node2: 23}}
    fist key node of departure, second node of arrival, value is the weight
    """
    
    def __init__(self):
        
        self.__nodes = dict()

    def __len__(self):
        return len(self.__nodes)

    def nodes(self):
        return list(self.__nodes.keys())
    
    def graph(self):
        return self.__nodes

    def __str__(self):
        ret = ""
        for n in self.__nodes:
            for edge in self.__nodes[n]:

                ret += "{} -- {} --> {}\n".format(str(n),
                                                  str(self.__nodes[n][edge]),
                                                  str(edge))
        return ret
    
    def insertNode(self, node):
        if node not in self.__nodes:
            self.__nodes[node] = {}

    def insertEdge(self, node1, node2, weight:int):
        if (node1 in self.nodes()) and (node2 in self.nodes()):
            # se entrambi sono del grafo
            self.__nodes[node1][node2] =  weight
            # print(f"Inserting edge from {node1} to {node2} with weight {weight} so: {self.__nodes[node1]}")
            # connecting node1 ---> node2
        else:
            raise NameError
            # return print("no such node in the graph")
    
    def deleteNode(self, node): # delete not connected nodes
        if node in self.__nodes:
            self.__nodes.pop(node)
        for n in self.__nodes:
            if node in self.__nodes[n]:
                self.__nodes[n].pop(node)

    def adjacent(self, node): # not needed
        # return all the nodes connected both way
        l = []
        if node in self.__nodes:
            for t in self.__nodes[node]:
                l.append(t)
                #   print(f"Appending {t}")
            for n in self.__nodes:
                for j in self.__nodes[n]:
                    if j == node:
                        l.append(n)
                        #   print(f"Appending {n}")
            return l
        else:
            return print("No such node in the graph")

    def adjacentEdge(self, node, incoming = False):
        # if incoming kept false return all the outgoing edges
        # append the dictionaries node value
        if node in self.__nodes:
            l = []
            if incoming:
                for n in self.__nodes:
                    for j in self.__nodes[n]:
                        if j == node:
                            l.append(n)
            else:
                # return the list of the connected nodes
                for t in self.__nodes[node].keys():
                    l.append(t)
            return l

    def edges(self):
        l = []
        for n in self.__nodes:
            for j in self.__nodes[n]:
                l.append(( n, j, self.__nodes[n][j]))
        return l
    
    def edge_weight(self, node1, node2):
        weight = self.__nodes[node1][node2]
        return weight
    
    def not_connect(self):
        # return all the nodes thet are not connected
        looser = []
        for n in self.nodes():
            if len(self.adjacent(n)) == 0:
                looser.append(n)
        return looser

    def DFS_path(self, node, seen = None , path = None):
        # return all the possible paths from a node
        # v is the node
        if seen is None: 
            seen = []
        if path is None: 
            path = [node]

        seen.append(node)

        paths = []
        for t in self.adjacentEdge(node): # edges uscenti da v
            if t not in seen:
                t_path = path + [t]
                #print(f"t_path{t_path}")
                paths.append(tuple(t_path))
                #print(f"paths {paths}")
                paths.extend(self.DFS_path(t, seen[:], t_path))
        #print(f"paths {paths}")
                
        real = []
        for p in paths:
            for n in p:
                if len(self.adjacentEdge(n)) == 0:
                    real.append(p)
        #print(f"real {real}")

        # return a list of list, where there are in tuple all the edges
        return real
    
    def f_max_score(self, paths):
        
        # paths is a list with pathways
        # the function select the one with the highest score -> sum of edges
        
        list_max_scores = []
        max_score = 0
        S = 0
        best_path = []
        for path in paths:
            for p in range(len(path)-1):
                # if p == (len(path)):
                #     continue
                S += self.edge_weight(path[p], path[p+1])
            if S == max_score:
                best_path.append(path)
                #list_max_scores.append(S)
            if S > max_score:
                max_score = S
                best_path = []
                best_path.append(path)
                list_max_scores = []
                #list_max_scores.append(S)
            S = 0
        #print(f"best paths {best_path}\nlist_max_scores {list_max_scores}")
        
        return best_path
    
def scores():
    """
    This function has the aim to gives the scores to che matches and mismatches.
    A few inputs will be request: match, mis-match and gap.
    Plese insert a negative value for the mis-match, if a positive one is inserted the algorithm
    will take quite a long time.
    """

    Score = {}

    print("Please enter the match score:")
    Score["match"] = int(input())

    if type(Score["match"]) != int:
        print("Sorry I need an integer number")
        Score["match"] = int(input())

    print("Please now insert the mis-match score:")
    Score["mis-match"] = int(input())

    if type(Score["mis-match"]) != int:
        print("Sorry I need an integer number")
        Score["mis-match"] = int(input())
    elif Score["mis-match"] > 0:
        print("This value should be negative")
        Score["mis-match"] = int(input())

    print("Please enter the gap score:")
    Score["left"] = Score["up"] = int(input())

    if type(Score["up"]) != int:
        print("Sorry I need an integer number")
        Score["left"] = Score["up"] = int(input())

    return Score

def new_direct(graph: Graph, i:int, j : int, scores_ : dict, direction = "diagonal", match = False):

    if direction == "diagonal":
        node1 = "_".join([str(x) for x in [i,j]])
        node2 = "_".join([str(x) for x in [i +1 ,j + 1]])
        if match:
            # print(node1, node2)
            graph.insertEdge(node2, node1, scores_["match"])
        else:
            # print(node1, node2)
            graph.insertEdge(node2, node1, scores_["mis-match"])

    if direction == "up":
        node1 = "_".join([str(x) for x in [i ,j +1]])
        node2 = "_".join([str(x) for x in [i +1 ,j + 1]])
        # print(node1, node2)
        graph.insertEdge(node2, node1, scores_["up"])

    if direction == "left":
        node1 = "_".join([str(x) for x in [i + 1 , j]])
        node2 = "_".join([str(x) for x in [i +1 ,j + 1]])
        #print(node1, node2)
        graph.insertEdge(node2, node1, scores_["left"])

def convert(pos_or_name, to_graph = True):
    if to_graph:
        node = "_".join([str(x) for x in pos_or_name])
        return node
    else:
        Pos = [int(x) for x in pos_or_name.split("_")]
        return Pos
    
def allig(list_nodes, max_score, seq1, seq2):

    """
    The positions are: j, i in this order
    in an example matrix, because these indices are the positions in the scoring matrix:
    [[],[]..] j the pos in the internal list, i pos of the list
    """

    seq_short = min(seq1, seq2)
    #print(seq_short)
    seq_long = max(seq1, seq2)
    #print(seq_long)


    indices = []
    cnt = 0
    for path in list_nodes:
        indices.append([])
        for n in path:
            indices[cnt].append(convert(n, to_graph = False))
        cnt += 1
    #print(f"indices {indices}")
        
    # return something like [ [ [j ,i  ] , [j,i ] ..] , [ [j,i ] , [j,i ] ] ...]
    all_allig = []

    for path in indices:

        short_sequence = []
        long_sequence = []
        prev_i = 0
        prev_j = 0
        

        for pos in path:
            
            i = pos[0] -1
            j = pos[1] -1
            if i < 0 or j < 0:
                continue
            if i == prev_i:
                #print(f"before pop short {short_sequence}")
                short_sequence.pop()
                #print(f"short after pop {short_sequence}")
                short_sequence.append("- ")
                short_sequence.append((seq_short[i] + " "))
                #print(f"short when gap {short_sequence}")
            if j == prev_j:
                #print(f"before pop {long_sequence}")
                long_sequence.pop()
                long_sequence.append("- ")
                #print(f"after pop {long_sequence}")
                long_sequence.append((seq_long[j] + " "))
                #print(f"long when gap {long_sequence}")
            if i != prev_i:
                short_sequence.append((seq_short[i] + " "))
                #print(f"short{short_sequence}")
            if j != prev_j:
                long_sequence.append((seq_long[j] + " "))
                #print(f"long {long_sequence}")
            prev_i = i
            prev_j = j 

        all_allig.append([long_sequence, "\n", short_sequence])

    true_val = []
    cnt = 0
    for i in all_allig:
        true_val.append([])
        for j in range(len(i)):
            if type(all_allig[cnt][j]) == list:
                all_allig[cnt][j].reverse()
                #   print(f"sequence {all_allig[cnt][j]}")
                true_val[cnt].append("".join(all_allig[cnt][j]))
        cnt += 1 
    print("\n")
    for i in true_val:
        for j in i:
            print(j)
        print("Score of the allignment: {}".format(max_score))
        print("\n\n")
                       
    return 

def scoring_matrix(seq1 : str, seq2 : str, scores: dict, G: Graph):

    """
    Given two sequences and a disctionary with the scores the function construct the scoring matrix
    adding at the position ij the related score in the three cases (macht, mis-match and gap).
    The function needs also a graph inizialized becase making the computation construct the trace back.
    """

    print("Creating the scoring matrix...")

    # need the two sequences, the croring parameters and an initialized graph
    seq_short = min(seq1, seq2)
    seq_long = max(seq1, seq2)

    len_min = len(seq_short)
    len_max = len(seq_long)

    matrix = np.zeros(shape = (len_min +1, len_max +1), dtype = np.int_)

    for i in range(len_min + 1):
        for j in range(len_max + 1):
            G.insertNode("_".join([str(x) for x in [i,j]]))

    max_values = [0]
    maxi_positions = [0]
    # position are y-x coordinates
    
    for i in range(len_min):
        
        for j in range(len_max):

            # now we construct the matrix
            Match = False

            if seq_short[i] == seq_long[j]:
                diagonal = matrix[i][j] + scores["match"]
                Match = True
            if seq_short[i] != seq_long[j]:
                diagonal = matrix[i][j] + scores["mis-match"]
            orizontal = matrix[i+1][j] +scores["left"]
            vertical = matrix[i][j+1] + scores["up"]

            # here we aer writing the scoring matrix
            matrix[i+1][j+1] = max(diagonal, vertical, orizontal, 0)
            val = matrix[i+1][j+1]

            # this is for the maximum and his position
            if matrix[i+1][j+1] > max(max_values):
                max_values = [matrix[i+1][j+1]]
                maxi_positions = [(i +1, j +1)]
            if (matrix[i+1][j+1] == max_values[0]) and ((i+1,j+1) != maxi_positions[0]):
                max_values.append(matrix[i+1][j+1])
                maxi_positions.append((i +1,j +1))

            # now we are preparing for the back tracking

            if val != 0:
                if val == diagonal :
                    new_direct(G, i, j, scores,match = Match)

                if val == orizontal :
                    new_direct(G, i, j, scores, direction = "left")

                if val == vertical :
                    new_direct(G, i, j, scores, direction = "up")
                    
                if (val == diagonal) and (val == orizontal) : 
                    new_direct(G, i, j, scores, match = Match)
                    new_direct(G, i, j, scores, direction = "left")

                if (val == diagonal) and (val == vertical) :
                    new_direct(G, i, j,scores, match= Match)
                    new_direct(G, i, j, scores,direction = "up")

                if (val == orizontal) and (val == vertical):
                    new_direct(G, i, j, scores, direction = "up")
                    new_direct(G, i, j, scores, direction="left")

                if (val == diagonal) and (val == orizontal) and ( val == vertical):
                    new_direct(G, i, j,scores, match = Match)
                    new_direct(G, i, j, scores,direction = "up")
                    new_direct(G, i , j, scores,direction = "left")

    print("The scoring matrix:\n{}".format(matrix))
    #print(maxi_positions)
    #print(max_values)
    print("Done structuring the scoring matrix and created the graph")
    #return the graph and the list with tha maximum positions in a tuple
    return  (maxi_positions, max_values[0])

def Smith_Waterman(sequence1, sequence2):
    
    """
    Take as input two string object and performe the local alignment
    """
    
    Scores = scores()
    Trace_Back_graph = Graph()
    temp = scoring_matrix(sequence1, sequence2, Scores, Trace_Back_graph)
    Start = temp[0]
    score_max = temp[1]
    starting_nodes = []    
    for s in Start:
        starting_nodes.append(convert(s))
    for node in starting_nodes:
        paths = Trace_Back_graph.DFS_path(node) # return all the paths from the starting node 
        bests_paths = Trace_Back_graph.f_max_score(paths) # return only the paths with maximum score
        allig(bests_paths, score_max, sequence1, sequence2)
        
    return 

def check(some:list):

    """
    Check if there are non sense letters, such as à,@ or numbers.
    """
    bb = True
    if len(some)== 0:
        bb = False
        return bb
    alphabet = ["Q","W","E","R","T","Y","U","I","O","P","L","K",
                "J","H","G","F","D","S","A","Z","X","C","V","B","N","M"]
    for j in some:
        for i in j:
            if i not in alphabet:
                bb = False
                continue
    return bb


obj = [obj.upper() for obj in [args.sequence_1, args.sequence_2] if args.sequence_1 != None]

if check(obj):
    Smith_Waterman(obj[0], obj[1])
else:
    print("Please insert a sequence:\n")
    seq1 = str(input()).upper()
    print("Please insert a second sequence:\n")
    seq2 = str(input()).upper()

    Smith_Waterman(seq1, seq2)
    
