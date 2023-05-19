from fractions import Fraction

    

def getRelevance(xdata, ydata, node):
    """
    Gather relevance for a given node.

    Parameters
    ----------
    node
        Node for which the relevance should be obtained.
    xdata
        xdata
    ydata
        data as np array
    """
    p1 = Fraction(xdata[(xdata[:,node]==1)& (ydata==1)].shape[0], xdata[(xdata[:,node]==1)].shape[0]) if xdata[(xdata[:,node]==1)].shape[0] != 0 else 0
    p2 = Fraction(xdata[(xdata[:,node]==1)& (ydata==0)].shape[0], xdata[(xdata[:,node]==1)].shape[0]) if xdata[(xdata[:,node]==1)].shape[0] != 0 else 0
    p3 = 1 -p1
    p4 = 1 -p2


    rel = (p1-p2)**2 + (p3-p4)**2
    #print(p1, p2, p3, p4, rel)
    return rel


