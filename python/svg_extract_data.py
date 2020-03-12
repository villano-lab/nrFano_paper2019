import numpy as np
import math
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def get_dimensions(doc):
    ymin, ymax, xmin, xmax = 4*(np.NaN,)
    
    for element in doc:
        # first see if there's an axis
        # add it if it's not a duplicate
        path = element.path
        for line in path:
            x_arr = (line.start.real, line.end.real, xmin, xmax)
            y_arr = (line.start.imag, line.end.imag, ymin, ymax)
            
            xmin = np.nanmin(x_arr)
            xmax = np.nanmax(x_arr)
            ymin = np.nanmin(y_arr)
            ymax = np.nanmax(y_arr)
      
    print (xmin, xmax, ymin, ymax)
    return xmin, xmax, ymin, ymax

def get_axes(doc, width, height): # edited
    """send in individual paths,
    and also largest x and y value among all paths"""
    yaxes = []
    xaxes = []
    tol = 0.05
    
    for element in doc:
        # first see if there's an axis
        # add it if it's not a duplicate
        path = element.path
        for line in path:
            isVertical = math.isclose(line.start.real, line.end.real)
            isHorizontal = math.isclose(line.start.imag, line.end.imag)
            
            if math.isclose(line.length(), width, rel_tol=tol) and isHorizontal: 
                if (len(xaxes) < 1) or all(line != axis for axis in xaxes):
                    #print ("xaxis isHorizontal, isVertical", isHorizontal, isVertical)
                    xaxes.append(line)
                    
            elif math.isclose(line.length(), height, rel_tol=tol) and isVertical:
                if (len(yaxes) < 1) or all(line != axis for axis in yaxes):
                    #print ("yaxis isHorizontal, isVertical", isHorizontal, isVertical)
                    yaxes.append(line)
    
    return (xaxes, yaxes)


def get_calib_points(doc, axes):
    yaxes, xaxes = axes
    calib_points = []
    calib_xlines = []
    calib_ylines = []
 
    for element in doc:
        # first see if there's an axis
        # if not, return
        path = element.path
        for line in path:
            # first see if it's a tick mark on the x-axis
            # test if it's vertical
            isVertical = math.isclose(line.start.real, line.end.real)
            # test if the starting y-coordinate is on any xaxis
            # can use axis.start.imag or axis.end.imag interchangeably
            isStartOnAxis = any(math.isclose(line.start.imag, axis.start.imag) for axis in xaxes)
            # test if the end y-coordinate is on any xaxis
            # can use axis.start.imag or axis.end.imag interchangeably
            isEndOnAxis = any(math.isclose(line.end.imag, axis.start.imag) for axis in xaxes)
            # test if the line is short
            isShort = line.length() < yaxes[0].length()/10
            
            if isVertical and (isStartOnAxis or isEndOnAxis) and isShort:
                calib_xlines.append(line)
                    
            # now see if it's a tick mark on the y-axis
            # test if it's horizontal
            isHorizontal = math.isclose(line.start.imag, line.end.imag)
            # test if the starting x-coordinate is on any yaxis
            # can use axis.start.real or axis.end.real interchangeably
            isStartOnAxis = any(math.isclose(line.start.real, axis.start.real) for axis in yaxes)
            # test if the end y-coordinate is on any xaxis
            # can use axis.start.imag or axis.end.imag interchangeably
            isEndOnAxis = any(math.isclose(line.end.real, axis.start.real) for axis in yaxes)
            # test if the line is short
            isShort = line.length() < xaxes[0].length()/10
            
            if isHorizontal and (isStartOnAxis or isEndOnAxis) and isShort:
                calib_ylines.append(line)
                #print("line length ", line.length())
    
    # need to deal with minor, major tick marks
    thresh = 0.5
    x_lengths = [line.length() for line in calib_xlines]
    x_lengths = np.sort(x_lengths)
    min_xlengths = [group.min() for group in np.split(x_lengths, np.where(np.diff(x_lengths) > thresh)[0]+1)]
    print (min_xlengths)

    y_lengths = [line.length() for line in calib_ylines]
    y_lengths = np.sort(y_lengths)
    min_ylengths = [group.min() for group in np.split(y_lengths, np.where(np.diff(y_lengths) > thresh)[0]+1)]
    print (min_ylengths)

    for line in calib_xlines:
        # test if the starting y-coordinate is on any xaxis
        # can use axis.start.imag or axis.end.imag interchangeably
        isStartOnAxis = any(math.isclose(line.start.imag, axis.start.imag) for axis in xaxes)
        # test if the end y-coordinate is on any xaxis
        # can use axis.start.imag or axis.end.imag interchangeably
        isEndOnAxis = any(math.isclose(line.end.imag, axis.start.imag) for axis in xaxes)
        # test if the line is a major tick
        isMajor = line.length() >= max(min_xlengths)
        
        if (isStartOnAxis or isEndOnAxis) and isMajor:
            if isStartOnAxis:
                calib_points.append((line.start.real, line.start.imag))
            elif isEndOnAxis:
                calib_points.append((line.start.real, line.end.imag))

    for line in calib_ylines:
        # test if the starting x-coordinate is on any yaxis
        # can use axis.start.real or axis.end.real interchangeably
        isStartOnAxis = any(math.isclose(line.start.real, axis.start.real) for axis in yaxes)
        # test if the end y-coordinate is on any xaxis
        # can use axis.start.imag or axis.end.imag interchangeably
        isEndOnAxis = any(math.isclose(line.end.real, axis.start.real) for axis in yaxes)
        # test if the line is short
        isMajor = line.length() >= max(min_xlengths)
        
        if (isStartOnAxis or isEndOnAxis) and isMajor:
            #print("line length ", line.length())
            if isStartOnAxis:
                calib_points.append((line.start.real, line.start.imag))
            elif isEndOnAxis:
                calib_points.append((line.end.real, line.start.imag))

    # call set to get rid of non-unique points
    return (list(set(calib_points)))
    
def calib(user_points, svg_points):
    xmin_user, ymin_user = np.amin(user_points, axis=0)
    
    # min returns ymax_svg because of the svg coordinate system
    xmin_svg, ymax_svg = np.amin(svg_points, axis=0)
    xmax_svg, ymin_svg = np.amax(svg_points, axis=0)

    # get the x-values from the svg file and user
    xcalib_svg = svg_points[np.isclose(svg_points[:,1], ymin_svg)][:,0]
    xcalib_user = user_points[np.isclose(user_points[:,1], ymin_user)][:,0]
    # make the values unique
    xcalib_svg = list(set(xcalib_svg))
    xcalib_user = list(set(xcalib_user))
    # sort the list in-place
    xcalib_svg.sort()
    xcalib_user.sort()
    #print(xcalib_svg, xcalib_user)
    xcalib_fn = get_calib_func(xcalib_user, xcalib_svg)
    
    # get the y-values from the svg file and user
    ycalib_svg = svg_points[np.isclose(svg_points[:,0], xmin_svg)][:,1]
    ycalib_user = user_points[np.isclose(user_points[:,0], xmin_user)][:,1]
    # make the values unique
    ycalib_svg = list(set(ycalib_svg))
    ycalib_user = list(set(ycalib_user))
    # sort the list in-place
    # note that because of svg coordinate system, small user value goes with large svg value
    ycalib_svg.sort(reverse=True)
    ycalib_user.sort()
    #print(ycalib_svg, ycalib_user)
    ycalib_fn = get_calib_func(ycalib_user, ycalib_svg)
    
    return xcalib_fn, ycalib_fn
    
    
def get_calib_func(user_arr, svg_arr):
    # save this for when there are actually double-calib points 
    # will choose the best fit
    svg_arr_list = []
    
    if len(user_arr) != len(svg_arr):
        r = np.average(np.diff(svg_arr))/2
        svg_arr_points = np.ones((len(svg_arr),2))
        svg_arr_points[:,0] = svg_arr[:]

        tree = cKDTree(svg_arr_points)
        rows_to_fuse = tree.query_pairs(r=r,output_type='ndarray') 
        for idx in rows_to_fuse.flatten():
            #print(idx)
            svg_arr_list.append(np.delete(svg_arr, idx))
    
    else:
        svg_arr_list.append(svg_arr)
        
    # fit a linear function to the arrays
    fit_goodness = []
    fit_arr = []
    for svg_arr in svg_arr_list:
        print(svg_arr, user_arr)
        fit, res, rank, sing, thresh = np.polyfit(svg_arr, user_arr, 1, full=True)
        #print("res: ", res)
        fit_goodness.append(res)
        fit_arr.append(np.poly1d(fit))
        
    min_idx = np.argmin(fit_goodness)
    fit_fn = fit_arr[min_idx]
        
    if False:
        plt.plot(svg_arr_list[min_idx], user_arr,'o')
        plt.plot(svg_arr_list[min_idx], fit_fn(svg_arr_list[min_idx]),'+')
        plt.show()
        
    return fit_fn
       

def get_calib_fn (doc, graph_points):
    xmin, xmax, ymin, ymax = get_dimensions(doc)

    width = xmax - xmin
    height = ymax - ymin
    #height = 147.37939999999998 - 2.5768999999999664
    #width = 175.0608 - 34.730800000000016

    xaxes, yaxes = get_axes(doc, width, height)
    print (xaxes, yaxes)

    svg_calibPoints = np.array(get_calib_points(doc, (xaxes, yaxes)))
    print("svg calibration points:")
    print(svg_calibPoints)

    #print (graph_points)
    xcalib_fn, ycalib_fn = calib(graph_points, svg_calibPoints)
    
    return xcalib_fn, ycalib_fn 

def get_points_from_path(path, t_arr, xcalib_fn, ycalib_fn):
    #print (path)
    point_arr = []
    for t in t_arr:
        point = (xcalib_fn(path.point(t).real), ycalib_fn(path.point(t).imag))
        point_arr.append(point)
        
    return np.array(point_arr)
