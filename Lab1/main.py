import monkdata as m
import dtree as d
import drawtree_qt5 as dt
import random
import matplotlib.pyplot as plt
'''
Assignment0:
We think that monk2 dataset is the hardest one to learn since it has all six variables 
and have the largest dataset. 
'''

'''
Assignment1:
'''
print("------------Assignment 1 ----------------")
ent1 = d.entropy(m.monk1)
ent2 = d.entropy(m.monk2)
ent3 = d.entropy(m.monk3)
print("Entropy for monk1:", ent1)
print("Entropy for monk2:", ent2)
print("Entropy for monk3:", ent3)

'''
Assignment2:
    Uniform distributions produce higher entropy, while non-uniform distributions can have less 
    entropy. Like in dice example. A fake die has less entropy than a perfect die. 
'''

'''
Assignment3: Calculate the average Gain 
'''
print("------------Assignment 3 ----------------")
for i in range(1, 4):
    for j in range(0, 6):
        if i == 1:
            print("{:.5f}".format(d.averageGain(m.monk1, m.attributes[j])), end=" ")
        elif i == 2:
            print("{:.5f}".format(d.averageGain(m.monk2, m.attributes[j])), end=" ")
        elif i == 3:
            print("{:.5f}".format(d.averageGain(m.monk3, m.attributes[j])), end=" ")
    print() #Newline

'''
Assignment4: 
    If the chosen set Sk is minimized then the total gain is maximized.
    What needs to be thought of when picking an attribute is to make the tree
    as balanced as possible
'''
'''
Assignment5: Build decision tree  
    Yes, our assumptions was indeed correct since we thought that monk2 dataset was 
    the hardest dataset to learn.
    Asking is done by taking a dataset and an attribute and ask what is the highest average 
    gain for the attribute to take a specific value (eg. attr 5 can take 1-4 values)  
    We do this for each value of the attribute 5 (value 1-4) 
'''
print("------------Assignment 5 (PART 1) ----------------")
ask1 = d.select(m.monk1, m.attributes[4], 1)
print("Most common ask1", d.mostCommon(ask1))
print(d.averageGain(ask1, m.attributes[0]),
      d.averageGain(ask1, m.attributes[1]),
      d.averageGain(ask1, m.attributes[2]),
      d.averageGain(ask1, m.attributes[3]),
      d.averageGain(ask1, m.attributes[4]),
      d.averageGain(ask1, m.attributes[5]))
ask2 = d.select(m.monk1, m.attributes[4], 2)
print("Most common ask2", d.mostCommon(ask2))
print(d.averageGain(ask2, m.attributes[0]),
      d.averageGain(ask2, m.attributes[1]),
      d.averageGain(ask2, m.attributes[2]),
      d.averageGain(ask2, m.attributes[3]),
      d.averageGain(ask2, m.attributes[4]),
      d.averageGain(ask2, m.attributes[5]))
ask3 = d.select(m.monk1, m.attributes[4], 3)
print("Most common ask3", d.mostCommon(ask3))
print(d.averageGain(ask3, m.attributes[0]),
      d.averageGain(ask3, m.attributes[1]),
      d.averageGain(ask3, m.attributes[2]),
      d.averageGain(ask3, m.attributes[3]),
      d.averageGain(ask3, m.attributes[4]),
      d.averageGain(ask3, m.attributes[5]))
ask4 = d.select(m.monk1, m.attributes[4], 4)
print("Most common ask4", d.mostCommon(ask4))
print(d.averageGain(ask4, m.attributes[0]),
      d.averageGain(ask4, m.attributes[1]),
      d.averageGain(ask4, m.attributes[2]),
      d.averageGain(ask4, m.attributes[3]),
      d.averageGain(ask4, m.attributes[4]),
      d.averageGain(ask4, m.attributes[5]))

t1 = d.buildTree(m.monk1, m.attributes);
t2 = d.buildTree(m.monk2, m.attributes);
t3 = d.buildTree(m.monk3, m.attributes);
print("------------Assignment 5 (PART 2) ----------------")
#After the decision trees are built use the check-function which measures the performance
#for each tree. Take 1 minus the result to get the error rate
print("Error rate of test data monk1 {:.5f}".format(1-d.check(t1, m.monk1test)))
print("Error rate of training data monk1 {:.5f}".format(1-d.check(t1, m.monk1)))
print("Error rate of test data monk2 {:.5f}".format(1-d.check(t2, m.monk2test)))
print("Error rate of training data monk2 {:.5f}".format(1-d.check(t2, m.monk2)))
print("Error rate of test data monk3 {:.5f}".format(1-d.check(t3, m.monk3test)))
print("Error rate of training data monk3 {:.5f}".format(1-d.check(t3, m.monk3)))


#dt.drawTree(t1) #Depth
#dt.drawTree(t2) # Broad
#dt.drawTree(t3)

'''
Assignment6: Pruning of the tree  
    When pruning nodes the tree becomes less complex
    where the bias increases and variance decreases.
    The idea is to find a balance in the variance/bias
    tradeoff by pruning it the right way (optimizing 
    the error rate/performance). 
'''
print("------------Assignment 7 ----------------")
def prune(monktrain, monkval):
    p_tree = d.buildTree(monktrain, m.attributes)
    new_performance = init_performance = d.check(p_tree, monkval)
    currentBest = True
    while currentBest:
        prunes = d.allPruned(p_tree)
        for prune in prunes:
            test_performance = d.check(prune, monkval)
            currentBest = False
            if test_performance > new_performance:
                p_tree = prune
                new_performance = test_performance
                currentBest = True
    return [init_performance, d.check(p_tree, monkval)]

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def displayAss7(dataset,name):
    frac_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pre_pruning = []
    post_pruning = []
    number_of_runs = 500
    for i in frac_list:
        performance_compare = [0,0]
        for j in range(0, number_of_runs):
            monk1train, monk1val = partition(dataset, i)
            res_performance = prune(monk1train, monk1val)
            performance_compare[0] += res_performance[0]
            performance_compare[1] += res_performance[1]
        performance_compare[0] = performance_compare[0]/number_of_runs
        performance_compare[1] = performance_compare[1] /number_of_runs
        pre_pruning.append(1-performance_compare[0])
        post_pruning.append(1-performance_compare[1])


    plt.plot(frac_list, post_pruning, label = "Post Pruning")
    plt.plot(frac_list, pre_pruning, label = "Before Pruning")
    #Graph legend
    plt.legend()
    # naming the x axis
    plt.xlabel('Fraction')
    # naming the y axis
    plt.ylabel('Error Rate')
    #plt title
    plt.title(name)
    #plt.show()

displayAss7(m.monk1, 'Monk 1')
displayAss7(m.monk3, 'Monk 3')
