from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def computeVariance(arrayList):
    for ele in arrayList:
        ele = float(ele)
    LEN = float(len(arrayList))
    array = np.array(arrayList)
    sum1 = array.sum()
    array2 = array * array
    sum2 = array2.sum()
    mean = sum1/LEN
    variance = sum2 / LEN - mean**2
    return variance

#node class
class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))

def kdtree(x,y,a,b,ax,point_list,depth=0):
    try:
        k=len(point_list[0]) #k dimension
    except IndexError as e: # if not point_list:
        return None

    axis = depth%k# change axis based on depth

    #change axis based on maximum variance
    #max_var=0
    #for i in range(k):
    #    ll=[]
    #    for t in point_list:
    #        ll.append(t[i-1])
    #    var=computeVariance(ll)
    #    if var>max_var:
    #        max_var=var
    #        axis=i-1

    # sort point list according to axis value
    point_list.sort(key=itemgetter(axis))

    # choose median to split to left and right
    median=len(point_list)//2

    #plot line
    if axis==1:
        ax.plot([x,a],[point_list[median][1],point_list[median][1]],color= "b")
    else:
        ax.plot([point_list[median][0],point_list[median][0]], [y,b], color="r")

    #find lower left corner and upper right corner of subtree
    if axis==0:
        left_x=x
        left_y=y
        right_x=point_list[median][0]
        right_y=y
        left_a=point_list[median][0]
        left_b=b
        right_a=a
        right_b=b
    else:
        left_x=x
        left_y=y
        right_x=x
        right_y=point_list[median][1]
        left_a =a
        left_b=point_list[median][1]
        right_a=a
        right_b=b

    # create node and construct subtrees
    return Node(
        location=point_list[median],
        left_child=kdtree(left_x,left_y,left_a,left_b,ax,point_list[:median],depth+1),
        right_child=kdtree(right_x,right_y,right_a,right_b,ax,point_list[median+1:],depth+1)
    )

def main():
    input_data = pd.read_csv("points.txt", names=['x', 'y'], sep=' ')
    point_list = list(zip(input_data['x'], input_data['y']))
    plt.figure('2d tree', figsize=(max(input_data['x'])+5, max(input_data['y'])+5))
    plt.xlim([0,max(input_data['x'])+5])
    plt.ylim([0, max(input_data['y'])+5])
    ax = plt.gca()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    x_list = [p[0] for p in point_list]
    y_list = [p[1] for p in point_list]
    ax.scatter(x_list, y_list, c='g', s=40, alpha=0.5)
    tree = kdtree(0,0,max(input_data['x'])+5,max(input_data['y'])+5,ax,point_list)
    print(tree)
    plt.show()

if __name__ == '__main__':
    main()
