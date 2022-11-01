#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import matplotlib
import itertools
import hdbscan

class headss_plotting():
    
    def __init__(self, N, show = False):
        self.N         = N # Return number of cubes
        self.matrix_size = N*4
        if show:
            self.view_splitting = self.plot_splitting()
            self.view_stitching = self.plot_stitching()

    def __repr__(self)  : return f"""Produces a 2D visualisation of the splitting and stitching \
 process with {self.N} base cuts"""
        
    def create_new_colormap(self, new_colors):
        '''Create a custom colormap using a list of input colors'''
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        import matplotlib.colors as mcolors

        viridis = cm.get_cmap('viridis', 256)
        custom = viridis(np.linspace(0, 1, len(new_colors)))

        for i in range(len(new_colors)):
            custom[i,:3] = mcolors.to_rgb(new_colors[i])
        newcmp = ListedColormap(new_colors)
        return newcmp

    def savefigs(self, filename, dpi = 450):
        plt.savefig(f'{filename}.jpeg',
                    dpi=100, bbox_inches = 'tight')
        plt.savefig(f'{filename}.png',
                    dpi=dpi, bbox_inches = 'tight')
        plt.savefig(f'{filename}.pdf',
                    dpi=dpi, bbox_inches = 'tight')
    
        
    def plotOutline(self, ax, lw = 3, lc = 'k', ls = '-' ):
        matrix_size = self.N*4
        for i in [0, self.N*4]:
            ax.plot([-.48, matrix_size-.53],[i-.5, i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([i-0.5, i-0.5], [- 0.5, matrix_size-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines

    def fourthConditions(self, matrix, i, j, k, value = 0):
        matrix[i:i+2, i:i+2] = value
        matrix[i:i+2, (j-2):k] = value
        matrix[(j-2):k, (j-2):k] = value
        matrix[(j-2):k, i:i+2] = value
        return matrix

    def fourthMatrix(self):
        empty = np.ones((self.N*4,self.N*4))
        empty[:2, :] = empty[-2:, :] = 0
        empty[:, :2] = empty[:, -2:] = 0
        for i in range(1,int((len(empty)+1)/2)):
            if i ==0:
                    j = i; k = None
            else: j = -i; k = -i
            if (i-2) %4 == 0:
                empty = self.thirdConditions(empty, i,j,k, value = 0)
            elif (i) %4 == 0:
                empty = self.thirdConditions(empty, i,j,k, value = 2)
        return empty
    
    def thirdConditions(self, matrix, i, j, k, value = 0):
        matrix[i:i+2, i:i+2] = value
        matrix[i:i+2, (j-2):k] = value
        matrix[(j-2):k, (j-2):k] = value
        matrix[(j-2):k, i:i+2] = value
        return matrix

    def thirdMatrix(self):
        empty = np.ones((self.N*4,self.N*4))
        for i in range(0,int((len(empty)+1)/2)):
            if i ==0:
                    j = i; k = None
            else: j = -i; k = -i
            if i %4 == 0:
                empty = self.thirdConditions(empty, i,j,k, value = 0)
            elif (i-2) %4 == 0:
                empty = self.thirdConditions(empty, i,j,k, value = 2)
        return empty
    
    def baseLines(self, ax, i, lw = 3, lc = 'k', ls = '-'):
        if i%4==0: # Draw base layer            
            ax.plot([-.48,self.matrix_size-.53],[i-.5, i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([i-0.5, i-0.5], [- 0.5, self.matrix_size-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            
    def secondLines(self, ax, i, lw = 3, lc = 'k', ls = '-'):
        if (i-2)%4==0: # Draw secondary layer
            ax.plot([+1.48, self.matrix_size-2.53],[i-.5, i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([i-.5, i-.5], [+ 1.5, self.matrix_size-2.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines 
            
    def thirdLines(self, ax, i, lw = 3, lc = 'k', ls = '-'):
        matrix_size = self.matrix_size
        if (i-4)%4==0 and i <= matrix_size/2: # Third layer long lines.
            ax.plot([i-0.5, i-0.5], [i-2-.5, matrix_size-i+2-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([matrix_size-i-0.5, matrix_size-i-0.5], 
                        [i-2-.5, matrix_size-i+2-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            lc = 'darkred'
            ax.plot([+ i-2-.5, matrix_size-i+2-.5], [i-0.5, i-0.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines)
            ax.plot([+ i-2-.5, matrix_size-i+2-.5], 
                        [matrix_size-i-0.5, matrix_size-i-0.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines

        if (i-2)%4 == 0 and i <= matrix_size/2: # Draw third layer (short lines)
            if i == matrix_size/2:
               k = i-4
            else: k = i
            ax.plot([matrix_size-.48,matrix_size-k-2-.48],[i-.5, i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([-.48,k+2-.48],[matrix_size-i-.5,matrix_size-i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([matrix_size-.48,matrix_size-k-2-.48],
                        [matrix_size-i-.5,matrix_size-i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([-.48,k+2-.48],[i-.5, i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            lc = 'darkred'
            ax.plot([i-.5, i-.5],[-.48,k+2-.48], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([i-.5, i-.5], [matrix_size-.48,matrix_size-k-2-.48], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([matrix_size-i-.5,matrix_size-i-.5],[-.48,k+2-.48], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([matrix_size-i-.5,matrix_size-i-.5],
                        [matrix_size-.48,matrix_size-k-2-.48],
                        c = lc, lw = lw, zorder=2, ls = ls) # vertical lines
            
    def fourthLines(self, ax, i, lw = 3, lc = 'k', ls = '-'):
        matrix_size = self.matrix_size
        if (i-2)%4==0 and i <= matrix_size/2+2: # Fourth layer long lines.
            if i == 2:
                k = i
            else: k = (i - 4)
            ax.plot([i-0.5, i-0.5], [k+2-.5, matrix_size-k-2-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([matrix_size-i-0.5, matrix_size-i-0.5], 
                        [k+2-.5, matrix_size-k-2-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            lc = 'darkred'
            ax.plot([+k+2-.5, matrix_size-k-2-.5], [i-0.5, i-0.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines)
            ax.plot([+ k+2-.5, matrix_size-k-2-.5], 
                        [matrix_size-i-0.5, matrix_size-i-0.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines)

        if (i-2)%4 == 0 and i <= matrix_size/2+2: # Draw Fourth layer (short lines)
            if i > matrix_size/2:
               k = i-4
            else: k = i
            ax.plot([matrix_size-2-.48,matrix_size-k-.48],[i-.5-2, i-.5-2], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([+2-.48,k-.48],[matrix_size-i-.5+2,matrix_size-i-.5+2], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([matrix_size-.48-2,matrix_size-k-.48],
                        [matrix_size-i+2-.5,matrix_size-i+2-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([-.48+2,k-.48],[i-2-.5, i-2-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            lc = 'darkred'
            ax.plot([i-2-.5, i-2-.5],[-.48+2,k-.48], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([i-2-.5, i-2-.5], [matrix_size-.48-2,matrix_size-k-.48], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([matrix_size-i-.5+2,matrix_size-i-.5+2],[-.48+2,k-.48], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            ax.plot([matrix_size-i-.5+2,matrix_size-i-.5+2],
                        [matrix_size-.48-2,matrix_size-k-.48],
                        c = lc, lw = lw, zorder=2, ls = ls) # vertical lines
            
    def plot_splitting(self, save = False, lw = 3):
        # Define color schemes
        new_colors1 = ['khaki','cadetblue','olivedrab', 'w']
        new_colors2 = ['w', 'green']
        new_colors3 = ['cadetblue','lightskyblue', 'w']
        new_colors4 = ['peru', 'peachpuff','w']
        newcmp1 = self.create_new_colormap(new_colors=new_colors1)
        newcmp2 = self.create_new_colormap(new_colors=new_colors2)
        newcmp3 = self.create_new_colormap(new_colors=new_colors3[::-1])
        newcmp4 = self.create_new_colormap(new_colors=new_colors4[::-1])
        cmaps = [newcmp1, newcmp2, newcmp3, newcmp4]
        alpha = [0.5,0.5,0.75, 0.75]

        fig, axes = plt.subplots(1, 4, figsize=(45, 15), sharex = False, sharey = False)
        axs = axes.ravel()

        base_layer = np.zeros((self.N*4,self.N*4))
        secondary = np.zeros((self.N*4,self.N*4))
        secondary[2:-2, 2:-2] = 1
        third = self.thirdMatrix()
        fourth = self.fourthMatrix()
        for i,matrix in enumerate([base_layer, secondary, third, fourth]):
            axs[i].imshow(matrix, cmap = cmaps[i], alpha = alpha[i])

        for i in range(self.matrix_size+1): # Draw box lines
            lc = 'k'
            self.baseLines(ax = axs[0], i = i, lc = lc, lw = lw, ls = '--')   
            self.secondLines(ax = axs[1], i=i, lw = lw, lc = 'k', ls = '--')
            self.thirdLines(ax = axs[2], i=i, lw = lw, lc = 'k', ls = '--')
            self.fourthLines(ax = axs[3], i=i, lw = lw, lc = 'k', ls = '--')
        
        titles = ['Base Layer', 'Secondary Layer', 'Tertiary Layer', 'Quaternary Layer']

        for i, ax in enumerate(axs): # Format each plot
                ax.set_title(titles[i], fontsize = 48)
                ax.set_xlim(-.6, self.matrix_size- .4)
                ax.set_ylim(-0.6, self.matrix_size-.4)
                ax.axis('off')
                self.plotOutline(ax= ax)
        plt.show()
        
    ############################################################
    # These functions are for the stitching section of this work. 
    ############################################################

    def tertiaryCubes(self, value, matrix, i,j,k, offset = 1):
        if (i-offset)%4 == 0 and i>2 and i <= self.matrix_size/2-1: # Add Tertiary sides
                value = 2;
                for p in range(4, int(self.matrix_size/2)-2, 4):
                    if i >= p:
                        matrix[i-2+4:i+4, offset+p:offset+p+2] = value
                        matrix[-(i+4):-(i-2+4), offset+p:offset+p+2] = value

                        matrix[i-2+4:i+4, -offset-p-2:-offset-p] = value
                        matrix[-(i+4):-(i-2+4), -offset-p-2:-offset-p] = value

                        matrix[offset+p:offset+p+2, i-2+4:i+4] = value
                        matrix[offset+p:offset+p+2, -(i+4):-(i-2+4)] = value

                        matrix[-offset-p-2:-offset-p, i-2+4:i+4] = value
                        matrix[-offset-p-2:-offset-p, -(i+4):-(i-2+4)] = value
        return matrix

    def centralCubes(self, value, matrix, i,j,k, offset = 1):
        if (i-offset)%4 == 0 and i>2 and i <= self.matrix_size - 4: # Add Tertiary sides
            for p in range(i-offset, self.matrix_size-4, 4):
                matrix[i:i+2, offset+p:offset+p+2] = value
                matrix[offset+p:offset+p+2, i:i+2] = value
        return matrix

    def addSides(self, value, matrix, i,j,k, offset = 1):
        if (i-offset)%4 == 0 and i>2 and i <= self.matrix_size - 2: # Add Tertiary sides
            matrix[0:3, i:i+2] = value
            matrix[-3:, (j-2):k] = value
            matrix[(j-2):k, -3:] = value
            matrix[i:i+2, 0:3] = value
        return matrix
    
    def stitchingLines(self, ax, i, lw = 3, lc = 'k', ls = '-', offset = 0):
        if (i-offset)%2==0 and i >2 and i < self.matrix_size-2: # Draw base layer            
            ax.plot([-.48, self.matrix_size-.5],[i-.5, i-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # horizontal lines
            ax.plot([i-0.5, i-0.5], [- 0.5, self.matrix_size-.5], c = lc, 
                     lw = lw, zorder=2, ls = ls) # vertical lines
            
    def increaseMatrix(self, matrix, value):
        matrix = matrix + value
        return matrix

    def getBaseMatrix(self):
        matrix = np.zeros((self.N*4,self.N*4))
        return matrix

    def baseCorners(self, value, matrix, i,j,k, offset = 1):
        if i == 0: # Add base corners
            matrix[i:i+3, i:i+3] = value
            matrix[i:i+3, (j-3):k] = value
            matrix[(j-3):k, (j-3):k] = value
            matrix[(j-3):k, i:i+3] = value
        return matrix
    
    def N2_stitching(self, save = False, fontsize = 16):
        """Manual N=2 stitching to correct the edge case error in 'plot_stitching'.
        """
        # Define color scheme
        new_colors = ['khaki','cadetblue','olivedrab']
        newcmp = self.create_new_colormap(new_colors=new_colors)
        matrix = np.random.randint(0,5, (self.matrix_size, self.matrix_size))

        plt.rcParams['figure.figsize'] = 5,5
        # set matrix
        pooling = [[0,0,0,1, 1,0,0,0],
                   [0,0,0,1, 1,0,0,0],
                   [0,0,0,1, 1,0,0,0],
                   [1,1,1,2, 2,1,1,1],

                   [1,1,1,2, 2,1,1,1],
                   [0,0,0,1, 1,0,0,0],
                   [0,0,0,1, 1,0,0,0],
                   [0,0,0,1, 1,0,0,0],]
        # plot matrix
        plt.imshow(pooling, cmap = newcmp, alpha = 0.5)

        # Add gridlines to show pixels
        for i in range(self.matrix_size+1):
            if i == 3 or i == 5:
                lw = 2; ls = '--'; lc = 'darkred'
            elif i == 2 or i == 6:
                lw = 2; ls = '-'; lc = 'dimgrey'
            else:
                lw = 2; lc = 'k'; ls = '-'

            if i not in [1,7]:
                plt.plot([-.48,self.matrix_size-.53],[i-.5, i-.5], c = lc, 
                         lw = lw, zorder=2, ls = ls) # horizontal lines
                plt.plot([i-0.5, i-0.5], [- 0.5, self.matrix_size-.5], c = lc, 
                         lw = lw, zorder=2, ls = ls) # vertical lines

        plt.xlim(-.6, self.matrix_size- .4)
        plt.ylim(-0.6, self.matrix_size-.4)
        plt.axis('off')

    def plot_stitching(self):
        if self.N == 2:
            self.N2_stitching()
        else:
            colors = ['khaki','olivedrab', 'cadetblue', 'peru']
            cmp = self.create_new_colormap(new_colors=colors)
            matrix = self.getBaseMatrix()
            matrix = self.increaseMatrix(matrix = matrix, value=3)

            fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex = False, sharey = False)

            for i in range(0,len(matrix)):
                if i ==0:
                        j = i; k = None
                else: j = -i; k = -i
                # Add base lines for reference to stitching figure.
                self.baseLines(ax = axs, i = i, lc = 'w', 
                           ls = '--', lw = 1)
                # Add lines around each region.
                self.stitchingLines(ax = axs, i = i, 
                                lc = 'k', ls = '-', lw = 1, offset = 1)
                # Base regions (Yellow)
                matrix = self.baseCorners(value = 0, matrix = matrix, i=i,k=k,j=j, 
                                    offset = 1,) # Base corners        
                matrix = self.addSides(value = 0, matrix = matrix, i=i,k=k,j=j, 
                                       offset = 1) # Base sides
                matrix = self.centralCubes(value = 0, matrix = matrix, i=i,k=k,j=j, 
                                           offset = 1) # Base centers
                # Secondary regions (Green)
                matrix = self.centralCubes(value = 1, matrix = matrix, i=i,k=k,j=j, 
                                    offset = 3)
                # Tertiary Regions (Blue)
                matrix = self.addSides(value = 2, matrix = matrix, i=i,k=k,j=j, 
                                    offset = 3) # Tertiary sides
                matrix = self.tertiaryCubes(value = 2, matrix = matrix, i=i,k=k,j=j, 
                                    offset = 1) # Tertiary centers

            # Add details to plot
            axs.set_xlim(-.6, self.matrix_size- .4)
            axs.set_ylim(-0.6, self.matrix_size-.4)
            axs.axis('off')
            self.plotOutline(ax= axs, lc = 'k') # Add black outline
            axs.imshow(matrix, cmap = cmp) # Plot
        plt.title(f'Stitching Map for N = {self.N} Base Cut')
        plt.show()        

class headss_regions(headss_plotting):
    '''Description
    KEYWORDS
    - split regions - The regions from the splitting of data. These regions overlap 
    to avoid edge effects.
    - Stitch regions - The regions created during the stitching of Split regions. These
    regions do not overlap and are uniform apart from the edge regions extending.
    '''

    import matplotlib.pylab as plt
    import pandas as pd
    import numpy as np
    import matplotlib
    import itertools

    def __init__(self, N, df, split_columns, plot = False):
        
        headss_plotting.__init__(self, N = N)
        self.split_columns   = split_columns
        self.df              = df
        if type(df) == type(None):
            print(f"Warning, no data passed, only illustration purposes available")
        else:
            self.N_regions          = self.getNregions() # Number of regions
            self.limits, self.step  = self.getLimits()   # Limits for splitting and stitching
            self.low_cuts           = self.getMinimums() # Low limit cuts for splitting
            self.high_cuts          = self.getMaximums() # High limit cuts for splitting
            self.split_regions      = self.getSplitRegions() # Get split regions [DataFrame]
            self.stitch_regions     = self.getStitchRegions() # Get Stitch regions [DataFrame]
            self.split_data         = self.splitData() # Get input data with region labels.
            if plot:
                self.view_regions   = self.draw_data_cuts()
                self.view_stitching = self.plot_stitching()

    def __repr__(self)  : return f"""Allows data to be cut using a {self.N}x{self.N} Base Grid\ 
 in {len(self.split_columns)} Dimensions resulting in {self.N_regions} Regions"""

    def getNregions(self):
        '''Returns number of regions needed.'''
        return (2*self.N-1)**len(self.split_columns)

    def getStep(self):
        df = self.df[self.split_columns]
        step = (df.max().values-df.min().values)/self.N
        return step

    def getLimits(self):
        step = self.getStep() # Get width of each cube
        params = self.df.describe() # get limits of each column
        mins = pd.DataFrame() # Create an empty dataframe to record minimum cut values
        for index, stepValue in enumerate(step):
            col = self.split_columns[index]
            row = 0
            for a in np.arange(params[col].loc['min'], 
                               params[col].loc['max']-stepValue+params[col].loc['max']/100, 
                               stepValue/2):
                row+=1
                mins.loc[row, col] = a
        limits = np.array(list(itertools.product(*mins.T.values)))
        return limits, step

    def getDataPandas(self, minimums):
        '''Cut data by region'''
        df = self.df
        for i, value in enumerate(minimums): # Iterate over axis.
            df = df[df[self.split_columns[i]].between(value, value+self.step[i])]
        return df

    def splitDataFrames(self,):
        '''Splits the data into relevant dataFrame. 
        Returns an array of pandas.DataFrames.'''
        split = np.zeros(self.N_regions, dtype = object)
        for i, mins in enumerate(self.limits[:]):
            split[i] = self.getDataPandas(minimums = mins)
        return split

    def getSplitRegions(self):
        '''Create a DataFrame of each (split) regions limits.'''
        regions = pd.DataFrame(np.hstack((self.limits, self.limits+self.step)),
                           columns = [i+'_mins' for i in self.split_columns] + 
                                     [i+'_max'  for i in self.split_columns])
        regions = regions.reset_index().rename({'index': 'region'}, axis = 1)
        return regions
    
    def combineListOfDataFrames(self, members, add_pos = True, pos_name = 'index'):
        """combines a list of pd.DataFrames into one DataFrame
        INPUT
             members           - list of DataFrames [list]
             add_pos = False   - adds an column of list position [bool]
            pos_name = 'index' - column name from 'add_pos' [str]"""

        members_df = pd.DataFrame()
        for i, tmp_df in enumerate(members):
            if add_pos:
                tmp_df[pos_name] = i
            members_df = pd.concat([members_df, tmp_df], ignore_index = True)
        return members_df
    
    def splitData(self):
        """Cut data defined by the splitting process in HEADSS
        INPUT (from self)
                 df - dataset name from known types [str]
            columns - columns to be split [list]
                  N - N for base layer splits [int] """
        
        # Get data regions for clustering
        df = self.splitDataFrames()
        df = self.combineListOfDataFrames(df, add_pos = True, pos_name = 'region')
        return df

    ############################################################
    # These functions are for the stitching section of this work. 
    ############################################################

    def getMinimums(self):
        '''Gets minimum value for each (stitching) region'''
        low = self.limits+self.step*0.25
        # Fix edge effects, iterated over all columns.
        for i, minimum in enumerate(np.min(self.limits, axis = 0)):
            low[:,i][low[:, i]==min(low[:,i])] = minimum
        return low

    def getMaximums(self):
        '''Gets minimum value for each (stitching) region'''
        high = self.limits+self.step*0.75
        # Fix edge effects, iterated over all columns.
        for i, maximum in enumerate(np.max(self.limits, axis = 0)):
            high[:,i][high[:, i]==max(high[:,i])] = maximum+self.step[i]
        return high

    def getStitchRegions(self):
        regions = pd.DataFrame(np.hstack((self.low_cuts, self.high_cuts)), 
                               columns = [i+'_mins' for i in self.split_columns]+
                                         [i+'_max'  for i in self.split_columns])
        regions = regions.reset_index().rename({'index': 'region'}, axis = 1)
        return regions

    def stitchDataPandas(self, low_cut, high_cut):
        '''Cut data by region'''
        df = self.df
        for i, value in enumerate(low_cut): # Iterate over columns
            df = df[df[self.split_columns[i]].between(value, high_cut[i])]
        return df

    def getClusterRegionsStitched(self):
        dfRegions = pd.DataFrame()
        high_cuts = self.high_cuts
        for i,value in enumerate(self.low_cuts):
        #     rgba = cmap(np.random.random(1))
            t = self.stitchDataPandas(low_cut = value, high_cut = high_cuts[i])
            dfRegions = pd.concat((dfRegions,t))
        return dfRegions
    
class headss_hdbscan(headss_regions):
    '''Performs the clustering process, updating this class allows the use of other 
    clustering algorithms.'''
        
    def __init__(self, df, N, split_columns, cluster_columns = None, 
                 cluster = True, min_cluster_size = 5, min_samples = None, 
                 cluster_method = 'eom', allow_single_cluster = False):
            headss_regions.__init__(self, df = df, N = N, split_columns = split_columns)
            
            if cluster:
                # input parameters
                self.split_data = self.split_data
                self.min_cluster_size = min_cluster_size
                self.min_samples = min_samples
                self.cluster_method = cluster_method
                self.allow_single_cluster = allow_single_cluster
                self.cluster_columns = self.getClusterColumns(cluster_columns, 
                                                              split_columns)
                # output np.array of pd.DataFrames containing clustering results.
                self.members = self.clusterRegions()
                
    def __repr__(self)  : return f"""Allows data to be cut using a {self.N}x{self.N} Base Grid\ 
                in {len(self.split_columns)} Dimensions resulting in {self.N_regions} Regions"""
                
    def getClusterColumns(self, cluster_columns, split_columns):
        '''Allows clustering on split columns if clustering columns not specified'''
        if type(cluster_columns) == type(None):
            cluster_columns = split_columns
        else: cluster_columns = cluster_columns
        return cluster_columns
            
    def HDBSCAN(self, df, starting_cluster = 0, fontsize = 16, 
                drop_non_grouped = True):
        """ Cluster objects and format the results into a single dataframe.
        Updating this function allows use of other clustering algorithms"""
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                    min_samples=self.min_samples,
                                    prediction_data=False,
                                    allow_single_cluster=self.allow_single_cluster,
                                    cluster_selection_method=self.cluster_method,
                                    gen_min_span_tree=False).fit(df[self.cluster_columns])

        # Object info 
        groups=clusterer.labels_
        df.loc[:,'group'] = groups # Add groups to data
        # Avoid cluster ids from previous regions
        df.loc[df['group']!=-1, 'group'] += starting_cluster
        if drop_non_grouped:
            df = df.loc[df.group!=-1, :] # Remove datapoints not in cluster
        return df
                
    def clusterRegions(self):
        '''Cluster each region by HDBSCAN with cluster labels in order continuing 
        from the previous region.

        INPUT
            clustered_regions - list of DataFrames containing clustered regions'''

        N_clusters = 0
        # create a space to collect clustering data
        members = np.zeros(self.N_regions, dtype = object)
        for i in range(self.N_regions):
            # isolate data from selected region
            tmp = self.split_data.loc[self.split_data.region == i][:]
            # Run clustering
            members[i] = self.HDBSCAN(df = tmp,
                    drop_non_grouped = True, 
                    starting_cluster = N_clusters)
            # Avoids overlap with region crossover labels
            N_clusters = members[i].group.max()+1
        return members
    
class headss_stitching(headss_hdbscan):
    """Performs the stitching of results using a clustering in the format from 
    headss_HDBSCAN, which is split using headss_splitting
    INPUT
        see previous classes***
        merge = False - Merge large clusters that span across multiple reigons [bool]
        *** headss_merge ***
    OUTPUT
        members_df - DataFrame of clusters kept after the stitching process"""
    def __init__(self, N, df, split_columns, cluster_columns = None, 
                 df_clustered = False, min_cluster_size = 5, stitch_regions = None,
                 min_samples = None, cluster_method = 'eom', 
                 allow_single_cluster = False,):
        if not df_clustered:
            headss_hdbscan.__init__(self, df = df, N = N, split_columns = split_columns, 
                                    cluster_columns = cluster_columns,
                                    cluster = True,
                                    min_cluster_size = min_cluster_size, 
                                    min_samples = min_samples, 
                                    cluster_method = cluster_method, 
                                    allow_single_cluster = allow_single_cluster)
        else:
            self.members = df_clustered
            self.split_columns = split_columns
            self.stitch_regions = stitch_regions
        # Output
        self.centers = self.getCenters(self.members)
        self.members_df = self.stitching()
        
    def __repr__(self)  : return f"""Allows data to be cut using a {self.N}x{self.N} Base Grid\ 
                 in {len(self.split_columns)} Dimensions resulting in {self.N_regions} Regions"""
                
    def cutMisplacedClusters(self, centers):
        '''Drop clusters who's centers occupy the incorrect region defined by 
        stitching_regions.'''

        res = pd.DataFrame()
        for index, center in enumerate(centers):
            # Iterate over all centers to check it lies within the stitching map.
            center = center[np.all([(center[col].between(
                                self.stitch_regions.loc[index][f'{col}_mins'], 
                                    self.stitch_regions.loc[index][f'{col}_max']))
                                        for i, col in enumerate(self.split_columns)], 
                                            axis = 0)]
            res = pd.concat([res,center], ignore_index = True)
        return res
                
    def calculateCenters(self, data):
        '''Calculate the centre points for each cluster in data'''

        groups = pd.DataFrame(data.group.unique(), columns = ['group'])\
                                    .sort_values(by = 'group')
        cols = self.split_columns.copy()
        cols.append('N')
        df = pd.DataFrame([], columns = cols)
        for i in groups.group:
            tmp = data[data.group==i]
            df.loc[i] = tmp[cols[:-1]].median()
            df.loc[i,'N'] = tmp.shape[0]
        df = df.reset_index().rename(columns = {'index':'group'})
        return df

    def getCenters(self, members):
        '''returns center points of all clusters across all split regions'''

        centers = np.zeros(self.N_regions, dtype = object)
        for index,data in enumerate(members):
            centers[index] = self.calculateCenters(data = data)
        return centers
    
    def stitching(self):
        # Get centers of each clusters
        centers = self.centers
        # Remove clusters with centers in the void area of a region.
        remaining_clusters = self.cutMisplacedClusters(centers)
        # Combine members to a single DataFrame covering all feature space
        members_df = self.combineListOfDataFrames(self.members)
        # Delete clusters with centers outside of the stitching boundaries
        members_df = members_df[members_df.group.isin(remaining_clusters.group.values)]
        return members_df
    
class headss_merge(headss_stitching):
       
    def __init__(self, N, df, split_columns, cluster_columns = None, df_clustered = False, 
                 stitch_regions = None, min_cluster_size = 5, min_samples = None, 
                 cluster_method = 'eom', allow_single_cluster = False, merge = False, 
                 total_threshold = 0.1, overlap_threshold = 0.5, minimum_members = 10):
        headss_stitching.__init__(self, df = df, N = N, split_columns = split_columns,
                                  cluster_columns  = cluster_columns,
                                  df_clustered     = df_clustered,
                                  stitch_regions   = stitch_regions,
                                  min_cluster_size = min_cluster_size, 
                                  min_samples      = min_samples, 
                                  cluster_method   = cluster_method, 
                                  allow_single_cluster = allow_single_cluster)
        if merge:
            self.minimum_members = minimum_members
            self.total_threshold = total_threshold
            self.overlap_threshold = overlap_threshold
            self.members_df = self.mergeClusters()
            
    def __repr__(self)  : return f"""Allows data to be cut using a {self.N}x{self.N} Base Grid\ 
             in {len(self.split_columns)} Dimensions resulting in {self.N_regions} Regions"""
            
    def describe_clusters(self, group_col = 'group'):
        """get description of each cluster leaving zero for out of bound 
        clusters for naming purposes.
        INPUT
            group_col = ['group']  - column that contains group labels
        OUTPUT
            cluster_info - list of DataFrames describing each cluster 
            returns 0 for dropped clusters from Stitching process"""
        df = self.members_df
        groups = df[group_col].unique()
        cluster_info = np.zeros(int(max(groups)+1), dtype = object)
        for index, group in enumerate(groups):
            tmp = df[df[group_col] == group]
            cluster_info[group] = tmp.describe()[self.split_columns]\
                                        .loc[['count', 'min', 'max']]
        return cluster_info
    
    def find_overlapping_clusters(self, cluster_info):
        """Identifies cluster pairs that have potential to overlap
        INPUT
            cluster_info - cluster limits from 'describe_clusters'
            columns = ['x', 'y'] - columns to search, ie those where splitting occured
        OUTPUT
            matches - DataFrame of cluster pairs that potentially overlap """

        matches = []
        for i, df1 in enumerate(cluster_info):
            # Check cluster has not been dropped
            if type(df1) != int:
                for j, df2 in enumerate(cluster_info):
                    if type(df2) != int and i != j:
                        overlap = np.zeros(len(self.split_columns))
                        # Iterate over all axis
                        for index,col in enumerate(self.split_columns): 
                            if df2.loc['min', col] < \
                                        df1.loc['min', col] < \
                                            df2.loc['max', col]:
                                overlap[index] = 1
                        if any(overlap) == 1:
                            matches.append((i,j))
        matches = pd.DataFrame(matches, columns = ['group1', 'group2'])
        return matches

    def get_cluster_oob_info(self, df, index, regions,):
        '''Returns info about the out of bounds (oob) area of the clustering region 
        i.e. the area not included in the final stitching'''
        cluster = df[df.group == index]
        region = cluster.region.values[0]
        limits = []
        for i in self.split_columns:
            limits.append([regions.loc[region,f'{i}_mins'], 
                      regions.loc[region,f'{i}_max']])
        return cluster, limits

    def get_cluster_oob_matches(self, cluster_index,
                                evaluate = 'best'):
        '''Returns points of a cluster points out of bounds (oob) of the clustering region
        i.e. the area not included in the final stitching'''
        # Get stitch limits
        regions = self.split_regions
        df = self.members_df
        # Isolate target clusters
        cluster1, limits1 = self.get_cluster_oob_info(df, cluster_index[0], 
                                                      regions)
        cluster2, limits2 = self.get_cluster_oob_info(df, cluster_index[1], 
                                                      regions)
        for i, value in enumerate(limits1): # Iterate over all axis
            if value != limits2[i]: # Only check overlapping axis
    #             Limit members to overlapping region
                cluster2 = cluster2[cluster2[self.split_columns[i]]>float(limits1[i][0])]
                cluster1 = cluster1[cluster1[self.split_columns[i]]<float(limits2[i][1])]
        if len(cluster1) <= self.minimum_members or len(cluster2) <= self.minimum_members:
            return 0, [0,0]
        # merge overlapping region to check fractional matches
        merged = cluster1.merge(cluster2, how = 'inner', on = self.split_columns)
        return merged.shape[0],[merged.shape[0]/cluster1.shape[0], \
               merged.shape[0]/cluster2.shape[0]]

    def check_merge_branches(self, cluster_merge):
        '''Ensures all clusters merge to the final cluster in a chain. Without this branched 
        chains can have two final nodes which do not merge'''
        record = []; add = []
        for i,j in cluster_merge.values:
            # Avoids repeating checks
            if j not in record:
                record.append(j) # Update checked list
                tmp = cluster_merge.loc[cluster_merge.group2 == j] # Get list of joined groups
                if tmp.shape[0]>1: # If potential chain is identified
                    # Get unique gorups
                    uni = np.unique(np.hstack((tmp['group1']\
                                                .unique(),(tmp['group2'].unique()))))
                    for i in uni:
                        # Avoid redunency only check larger cluster numbers.
                        for j in uni[uni>i]: 
                            if i!=j: # Do not merge cluster to itself
                                add.append([max(i,j), min(i,j)])
        add = pd.DataFrame(add, columns = ['group1', 'group2']).drop_duplicates()
        return cluster_merge.merge(add, how = 'outer')

    def check_cluster_merge(self, matches):
        '''Checks if two clusters should merge
        INPUT
            df                - dataframe of cluster members
            overlap_threshold - fraction of members that overlap from the region of overlap
            total_threshold   - fraction of members that overlap from the total clusters
            minimum_members   - minimum overlap size for merging
        OUTPUT
            DataFrame containing the clusters to merge'''

        cluster_merge = []
        df = self.members_df
        for (i,j) in matches[:].values: # Remove :10 limit for full run
            tmp1 = df[df.group == i]
            tmp2 = df[df.group == j]
            N_merged, perc_merged = self.get_cluster_oob_matches(cluster_index = [i,j])
            if len(tmp1) == 0 or len(tmp2) == 0:
                continue
            else:
                perc_overlap = [N_merged/len(tmp1),N_merged/len(tmp2)]
                if max(perc_merged) > self.overlap_threshold and \
                                max(perc_overlap) > self.total_threshold:
                    cluster_merge.append([max(i,j), min(i,j)])
        cluster_merge = pd.DataFrame(cluster_merge, columns = ['group1', 'group2'])\
                                    .drop_duplicates()
        return self.check_merge_branches(cluster_merge = cluster_merge)
    
    def merge_overlapping_clusters(self, merge_list):
        '''iterate over merge list until cluster numbers stabalise'''
        df = self.members_df
        N_clusters = 0; N_clusters_pre = 1
        # Iterate until all chains are followed through
        while N_clusters != N_clusters_pre:
            N_clusters_pre = len(df.group.unique())
            for k in merge_list.values:
                i,j = min(k), max(k)
                # Move all clusters to max label to directionise the merges.
                df.loc[df["group"] == j, 'group'] = i
            N_clusters = len(df.group.unique())
        return df
    
    def mergeClusters(self):
        """Perform the final merge of overlapping clusters
        INPUT
            overlap_threshold = 0.5 - fraction of mutual members in the overlapping region
              total_threshold = 0.1 - fraction of mutual members in the total cluster
              minimum_members =  10 - minimum members in the overlapping region
        OUTPUT
            returns DataFrame with merged clusters showing a single label"""
        df = self.members_df
        cluster_info = self.describe_clusters()
        matches = self.find_overlapping_clusters(cluster_info = cluster_info[:])
        cluster_merges = self.check_cluster_merge(matches = matches)
        return self.merge_overlapping_clusters(merge_list = cluster_merges).drop_duplicates()
