import numpy as np
import argparse
import os.path as path
import Corrfunc
from Corrfunc._countpairs import countpairs

parser = argparse.ArgumentParser()
parser.add_argument('--critfile', type=str, required=True)
parser.add_argument('--galfile', type=str, required=True)
parser.add_argument('--binfile', type=str, required=True)
parser.add_argument('--boxsize', type=str, required=True)
parser.add_argument('--outfile1', type=str, required=True) #saddle-saddle (here and below saddle is referring to type-2)
parser.add_argument('--outfile2', type=str, required=True) #saddle-node
parser.add_argument('--outfile3', type=str, required=True) #saddle-galaxy
parser.add_argument('--outfile4', type=str, required=True) #node-node
parser.add_argument('--outfile5', type=str, required=True) #galaxy-galaxy
args = parser.parse_args()

#****************************************************************************************
#****************************************************************************************
#******************** Reading in critical point and galaxy positions ********************
#****************************************************************************************
#****************************************************************************************

dtype = [('X0', 'f4'), ('X1', 'f4'), ('X2', 'f4'),('value', 'f4'),
         ('type', 'i4'), ('pair_id', 'i4'), ('boundary', 'i4')]

crits = np.genfromtxt(args.critfile, dtype=dtype, comments='#')

#maskS = np.where( (crits['type']==2) & (crits['X0']>0.0) & (crits['X1']>0.0) & (crits['X2']>0.0))
#maskN = np.where( (crits['type']==3) & (crits['X0']>0.0) & (crits['X1']>0.0) & (crits['X2']>0.0))

maskS = crits['type']==2
maskN = crits['type']==3

saddles = crits[maskS]
nodes   = crits[maskN]

Ns = float(len(saddles))
print(Ns)
Nn = float(len(nodes))
print(Nn)

saddles_pos = np.vstack([saddles['X0'], saddles['X1'], saddles['X2']]).T
nodes_pos   = np.vstack([  nodes['X0'],   nodes['X1'],   nodes['X2']]).T
                         
gals  = np.genfromtxt(args.galfile)

Ng = float(len(gals))
print(Ng)

gals_pos = np.vstack([gals[:,2], gals[:,0], gals[:,1]]).T #axis flipping, not certain this is correct

print(min(saddles_pos[:,0]))
print(min(saddles_pos[:,1]))
print(min(saddles_pos[:,2]))

print(min(nodes_pos[:,0]))
print(min(nodes_pos[:,1]))
print(min(nodes_pos[:,2]))

#***************************************************************************
#***************************************************************************
#******************** Calculating correlation functions ******************** 
#***************************************************************************
#***************************************************************************

boxsize = float(args.boxsize)
print(boxsize)
nthreads = 4
binfile = path.abspath(str(args.binfile))

results_SS = countpairs(1, nthreads, binfile, boxsize=(boxsize, boxsize, boxsize), X1=saddles_pos[:,0].astype(np.float32), Y1=saddles_pos[:,1].astype(np.float32), Z1=saddles_pos[:,2].astype(np.float32), periodic=1, verbose=True)[0]
results_SN = countpairs(0, nthreads, binfile, boxsize=(boxsize, boxsize, boxsize), X1=saddles_pos[:,0].astype(np.float32), Y1=saddles_pos[:,1].astype(np.float32), Z1=saddles_pos[:,2].astype(np.float32), X2=nodes_pos[:,0].astype(np.float32), Y2=nodes_pos[:,1].astype(np.float32), Z2=nodes_pos[:,2].astype(np.float32), periodic=1, verbose=True)[0]
results_SG = countpairs(0, nthreads, binfile, boxsize=(boxsize, boxsize, boxsize), X1=saddles_pos[:,0].astype(np.float32), Y1=saddles_pos[:,1].astype(np.float32), Z1=saddles_pos[:,2].astype(np.float32), X2=gals_pos[:,0].astype(np.float32),  Y2=gals_pos[:,1].astype(np.float32),  Z2=gals_pos[:,2].astype(np.float32), periodic=1, verbose=True)[0]
results_NN = countpairs(1, nthreads, binfile, boxsize=(boxsize, boxsize, boxsize), X1=nodes_pos[:,0].astype(np.float32),   Y1=nodes_pos[:,1].astype(np.float32),   Z1=nodes_pos[:,2].astype(np.float32), periodic=1, verbose=True)[0]
results_GG = countpairs(1, nthreads, binfile, boxsize=(boxsize, boxsize, boxsize), X1=gals_pos[:,0].astype(np.float32),    Y1=gals_pos[:,1].astype(np.float32),    Z1=gals[:,2].astype(np.float32), periodic=1, verbose=True)[0]

xi_SS = []
xi_SN = []
xi_SG = []
xi_NN = []
xi_GG = []
rmin = []
rmax = []

i = 0
while i < len(results_SS):
  RR_SS = 0.5*Ns*Ns*(4./3.)*np.pi*(results_SS[i][1]**3.0 - results_SS[i][0]**3.0) / boxsize**3.0
  RR_SN = Ns*Nn*(4./3.)*np.pi*(results_SN[i][1]**3.0 - results_SN[i][0]**3.0) / boxsize**3.0
  RR_SG = Ns*Ng*(4./3.)*np.pi*(results_SG[i][1]**3.0 - results_SG[i][0]**3.0) / boxsize**3.0
  RR_NN = 0.5*Nn*Nn*(4./3.)*np.pi*(results_NN[i][1]**3.0 - results_NN[i][0]**3.0) / boxsize**3.0
  RR_GG = 0.5*Ng*Ng*(4./3.)*np.pi*(results_GG[i][1]**3.0 - results_GG[i][0]**3.0) / boxsize**3.0
  print(results_GG[i])
  
  xi_SS.append((results_SS[i][3]/RR_SS) - 1.)
  xi_SN.append((results_SN[i][3]/RR_SN) - 1.)
  xi_SG.append((results_SG[i][3]/RR_SG) - 1.)
  xi_NN.append((results_NN[i][3]/RR_NN) - 1.)
  xi_GG.append((results_GG[i][3]/RR_GG) - 1.)
  rmin.append(results_SS[i][0])
  rmax.append(results_SS[i][1])
  
  i += 1

np.savetxt(args.outfile1, np.transpose(np.array([rmin, rmax, xi_SS])))
np.savetxt(args.outfile2, np.transpose(np.array([rmin, rmax, xi_SN])))
np.savetxt(args.outfile3, np.transpose(np.array([rmin, rmax, xi_SG])))
np.savetxt(args.outfile4, np.transpose(np.array([rmin, rmax, xi_NN])))
np.savetxt(args.outfile5, np.transpose(np.array([rmin, rmax, xi_GG])))

