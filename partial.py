import torch
import numpy as np
import subprocess

TYPE = np.complex128

class PWD:
    def __init__(self , momp , momw , xp , xw , restrict = None):
        self.pp = np.copy(momp)
        self.pw = np.copy(momw)
        self.xp = np.copy(xp)
        self.xw = np.copy(xw)

        import fs

        # array with integrands for h
        self.hfuncs = {}
        for k in fs.fs.__dict__.keys():
            # n : (l' , l , s , j , number of operator)
            n = tuple(map(int , k[5:].split("_")))
            f = np.vectorize(fs.fs.__dict__[k])
            self.hfuncs.update({n : f})

        # list of pwd numbers
        self.nums = []
        for k in self.hfuncs.keys():
            # k : (l' , l , s , j , number of operator)
            ll , l , s , j , io = k
           
            if(restrict is None):
                if(not (ll , s , j) in self.nums):
                    self.nums.append((ll , s , j))
                if(not (l , s , j) in self.nums):
                    self.nums.append((l , s , j))
            elif(isinstance(restrict , list)):
                if(not (ll , s , j) in self.nums):
                    if((ll , s , j) in restrict):
                        self.nums.append((ll , s , j))
                if(not (l , s , j) in self.nums):
                    if((l , s , j) in restrict):
                        self.nums.append((l , s , j))
            else:
                if(not (ll , s , j) in self.nums):
                    if((ll + s + restrict) % 2 == 1):
                        self.nums.append((ll , s , j))
                if(not (l , s , j) in self.nums):
                    if((l + s + restrict) % 2 == 1):
                        self.nums.append((l , s , j))

            if(len(self.nums) != 0):
                self.lmax = max([x[0] for x in self.nums])
                self.smax = max([x[1] for x in self.nums])
                self.jmax = max([x[2] for x in self.nums])

        self.nums.sort(key = lambda x : x[2] * (self.smax + 1) * (self.lmax + 1) + x[0] * (self.smax + 1) + x[1])

        self.lsj = {}
        for i in range(len(self.nums)):
            self.lsj.update({i : self.nums[i]})

        # ha
        self.has = {}
        for k in self.hfuncs.keys():
            # k : (l' , l , s , j , number of operator)
            ha = self.hfuncs[k](self.pp[: , None , None] , self.pp[None , : , None] , self.xp[None , None , :])
            self.has.update({k : ha})

        self.m = np.zeros(
                    (
                        len(self.nums),
                        self.pp.shape[0] , 
                        len(self.nums),
                        self.pp.shape[0] , 
                        self.xp.shape[0] ,
                        6 
                    ),
                    dtype = np.complex128
                    )

        keys = self.has.keys()
        for io in range(6):
            ir = 0
            for r in self.nums:
                ic = 0
                for c in self.nums:
                    ll , ss , jj = r
                    l , s , j = c
                    if(ss == s and jj == j and ((ll , l , s , j , io + 1) in keys)):

                        self.m[
                                ir,
                                : ,
                                ic,
                                : ,
                                : ,
                                io
                        ] = self.has[(ll , l , s , j , io + 1)]

                    ic += 1
                ir += 1

    def getTorchTensor(self , device = 'cpu' , requires_grad = False):
        t = torch.from_numpy(self.m)
        if(requires_grad):
            t.requires_grad = True
        if(device != 'cpu'):
            t.to_device(device)
        return t 

    def getTorchWeightsX(self , device = 'cpu' , requires_grad = False):
        w = torch.from_numpy(self.xw)
        if(requires_grad):
            w.requires_grad = True
        if(device != 'cpu'):
            w.to_device(device)
        return w 

    def getTorchWeightsP(self , device = 'cpu' , requires_grad = False):
        w = torch.from_numpy(self.pw)
        if(requires_grad):
            w.requires_grad = True
        if(device != 'cpu'):
            w.to_device(device)
        return w 

    def getTorchPointsX(self , device = 'cpu' , requires_grad = False):
        w = torch.from_numpy(self.xp)
        if(requires_grad):
            w.requires_grad = True
        if(device != 'cpu'):
            w.to_device(device)
        return w 

    def getTorchPointsP(self , device = 'cpu' , requires_grad = False):
        w = torch.from_numpy(self.pp)
        if(requires_grad):
            w.requires_grad = True
        if(device != 'cpu'):
            w.to_device(device)
        return w 

    def getTorchInput(self,  fproc , device = 'cpu' , requires_grad = True):
        # fa

        import tempfile

        fpp = tempfile.NamedTemporaryFile(mode = "w" , delete = False)
        fpp.write(str(self.pp.shape[0]) + "\n")
        for p in self.pp:
            fpp.write(str(p) + "\n")
        fpp.close()

        fxp = tempfile.NamedTemporaryFile(mode = "w" , delete = False)
        fxp.write(str(self.xp.shape[0]) + "\n")
        for x in self.xp:
            fxp.write(str(x) + "\n")
        fxp.close()

        inp = np.zeros((self.pp.shape[0] , self.pp.shape[0] , self.xp.shape[0] , 6) , dtype = np.complex128)

        for ii in range(6):
            proc = subprocess.run([fproc , fpp.name , fpp.name , fxp.name , str(ii + 1)] , capture_output = True)
            fa = np.array(list(map(lambda l : float(l.strip()) , proc.stdout.decode('utf8').strip().split('\n'))))
            fa = fa.reshape((self.pp.shape[0] , self.pp.shape[0] , self.xp.shape[0]))
            inp[: , : , : , ii] = fa
            #fas.append(fa)

        import os
        os.remove(fpp.name)
        os.remove(fxp.name)

        m = torch.from_numpy(inp)
        if(requires_grad):
            m.requires_grad = True
        if(device != 'cpu'):
            m.to_device(device)

        return m

