#!/usr/bin/env python3

import numpy as np
import partial
import torch
import matplotlib.pyplot as plt

import os
import sys
import argparse

import shutil

def log(dir , message):
    with open(os.path.join(dir , "log") , 'a') as f:
        f.write(message + "\n")

def getR1(pzero , hwff , pp , pw , m , maxpval):
    logipi = np.log((maxpval + pzero) / (maxpval - pzero)) - (1j * np.pi)
    pzerok = torch.sum((pzero * pzero * pw[:-1]) / ((pzero * pzero - pp * pp)[:-1]))
    
    eh0 = pw * pp * pp / (pzero * pzero - pp * pp)
    eh0[-1] = 0.0
    
    a                  = -hwff *                 m * eh0[None , None , None , :]
    a[: , : , : , -1] +=  hwff[: , : , : , -1] * m * pzerok
    a[: , : , : , -1] -=  hwff[: , : , : , -1] * m * 0.5 * pzero * logipi

    return a

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description = "T matrix calculation with error propagation.")
    parser.add_argument("dir" , help = "Output directory.")
    parser.add_argument("pzero" , help = "p0" , type = float)
    parser.add_argument("dnum" , help = "discrete number for initial state" , type = int)
    parser.add_argument("mnum" , help = "momentum number for initial state" , type = int)
    parser.add_argument("--dnums" , help = "discrete numbers for restrict" , action = "append")
    parser.add_argument("--maxpval" , help = "Maximum momentum in fm^-1." , type = float , default = 10.0)
    parser.add_argument("--ppts" , help = "Number of points for momenta." , type = int , default = 64)
    parser.add_argument("--xpts" , help = "Number of points for x." , type = int , default = 64)
    parser.add_argument("--relerror" , help = "Relative error for scalar functions" , type = str , default = "0 0 0 0 0 0")
    parser.add_argument("--ntesterror" , help = "Number of samples to test error" , type = int)
    parser.add_argument("--forcepath" , help = "Path to program calculating scalar functions" , default = "./ff1")
    args = parser.parse_args()
  
    dnums = None
    if(args.dnums is not None):
        dnums = list(map(lambda x : tuple(map(int , x.split())) , args.dnums))

    if(os.path.isdir(args.dir)):
        sys.stderr.write("Directory " + args.dir + " exists. Exiting.\n")
        sys.exit(1)

    os.mkdir(args.dir)

    with open(os.path.join(args.dir , "cmd") , "w") as f:
        f.write(" ".join(sys.argv))
    shutil.copyfile(os.path.realpath(__file__) , os.path.join(args.dir , "test_scattering.py"))


    # relative errors
    rer = np.array(list(map(lambda x : float(x) , args.relerror.split())))

    # points and weights
    log(args.dir , "getting points and weights ...")

    pp , pw = np.polynomial.legendre.leggauss(args.ppts)
    pp = (pp + 1.0) * 0.5 * args.maxpval
    pw = pw * 0.5 * args.maxpval

    pp = np.append(pp , args.pzero)
    pw = np.append(pw , 0.0)

    xp , xw = np.polynomial.legendre.leggauss(args.xpts)

    log(args.dir , "saving points and weights for momenta and x to: pp.npy, pw.npy, xp.npy , xw.npy ...")
    np.save(os.path.join(args.dir , "pp.npy") , pp)
    np.save(os.path.join(args.dir , "pw.npy") , pw)
    np.save(os.path.join(args.dir , "xp.npy") , xp)
    np.save(os.path.join(args.dir , "xw.npy") , xw)

    # pwd data
    log(args.dir , "calculating pw data ...")

    pwData = partial.PWD(pp , pw , xp , xw , restrict = dnums)

    # integrands from equation (7) from [Eur. Phys. J. A 43 241-250 (2010)] 
    h = pwData.getTorchTensor()

    # torch.tensor weights for x integration
    w = pwData.getTorchWeightsX()
    
    # values of the scalar functions
    log(args.dir , "getting scalar function values ...")
    
    ff = pwData.getTorchInput(args.forcepath)
    ff = ff.to(torch.float64).detach()
    ff.requires_grad = True
    fff = ff.to(torch.complex128)

    # calculating errors
    ffstd = ff.abs() * torch.from_numpy(rer[None , None , None , :])

    # i - discrete number of the partal wave in the final state
    # j - numbers momentum in the final state
    # k - discrete number of the partial wave in the inital state
    # l - numbers momentum in the initial state
    log(args.dir , "calculating pw decomposition ...")

    hwff = torch.einsum('ijklmn , m , jlmn -> ijkl' , h , w , fff)

    if(True):
        with open("check_hwff" , "w") as f:
            for i in range(hwff.shape[0]):
                for j in range(hwff.shape[1]):
                    for k in range(hwff.shape[2]):
                        for l in range(hwff.shape[3]):
                            f.write(
                                    str(i) + " " + 
                                    str(j) + " " + 
                                    str(k) + " " + 
                                    str(l) + " " + 
                                    str(hwff[i , j , k , l].real.item()) + " " + 
                                    str(hwff[i , j , k , l].imag.item()) + "\n"
                            )

    # nucleon mass fm^-1
    nucm = 0.5 * (938.272 + 939.565) / 197.328

    # torch.tensor momentum points and weights 
    pp_tt = pwData.getTorchPointsP()
    pw_tt = pwData.getTorchWeightsP()
    log(args.dir , "saving torch tensor points and weights for momenta: pp_tt.pt, pw_tt.pt ...")
    torch.save(pp_tt , os.path.join(args.dir , "pp_tt.pt"))
    torch.save(pw_tt , os.path.join(args.dir , "pw_tt.pt"))

    r1 = getR1(args.pzero , hwff , pp_tt , pw_tt , nucm , args.maxpval)

    if(True):
        with open("check_r1" , "w") as f:
            for i in range(hwff.shape[0]):
                for j in range(hwff.shape[1]):
                    for k in range(hwff.shape[2]):
                        for l in range(hwff.shape[3]):
                            f.write(
                                    str(i) + " " + 
                                    str(j) + " " + 
                                    str(k) + " " + 
                                    str(l) + " " + 
                                    str(r1[i , j , k , l].real.item()) + " " + 
                                    str(r1[i , j , k , l].imag.item()) + "\n"
                            )

    r1 = r1.reshape(
                    (
                        hwff.shape[0] * hwff.shape[1] , 
                        hwff.shape[2] * hwff.shape[3]
                    )
                )

    r1 = torch.eye(r1.shape[0]) + r1

    res = torch.linalg.solve(r1 , hwff[: , : , args.dnum , args.mnum].reshape(hwff.shape[0] * hwff.shape[1]))

    res = res.reshape((hwff.shape[0] , hwff.shape[1]))

    log(args.dir , "saving torch tensor result to res.pt ...")
    torch.save(res , os.path.join(args.dir , "res.pt"))
    
    log(args.dir , "saving result to resre, resim ...")
    with open(os.path.join(args.dir , "resre") , "w") as f:
        for i in range(hwff.shape[0]):
            for j in range(hwff.shape[1]):
                f.write(str(res[i , j].real.item()) + " ")
            f.write("\n")
    with open(os.path.join(args.dir , "resim") , "w") as f:
        for i in range(hwff.shape[0]):
            for j in range(hwff.shape[1]):
                f.write(str(res[i , j].imag.item()) + " ")
            f.write("\n")
    
    log(args.dir , "saving momenta to pppp ...")
    with open(os.path.join(args.dir , "pppp") , "w") as f:
        for i in range(len(pp)):
            f.write(str(i) + " " + str(pp[i]) + " " + str(pw[i]) + "\n")

    log(args.dir , "saving discrete numbers to num ...")
    with open(os.path.join(args.dir , "num") , "w") as f:
        for i in range(len(pwData.nums)):
            f.write(str(i) + " " + str(pwData.nums[i]) + "\n")

    b = res[args.dnum ,  -1]

    s = 1.0 - 1.0j * args.pzero * nucm * torch.pi * b

    arg_s = 0.5 * torch.arctan2(s.imag , s.real) * 180 / torch.pi

    with open(os.path.join(args.dir , "arg_s") , "w") as f:
        f.write(str(arg_s.item()))

    arg_s.backward(retain_graph = True)

    log(args.dir , "saving gradient to ffgrad.py")
    torch.save(ff.grad , os.path.join(args.dir , "ffgrad.pt"))

    var = torch.sum((ffstd * ff.grad) * (ffstd * ff.grad))
    varsqrt = torch.sqrt(var)

    with open(os.path.join(args.dir , "varsqrt") , "w") as f:
        f.write(str(varsqrt.item()))

    for iii in range(args.ntesterror):
        log(args.dir , "iteration " + str(iii) + " of " + str(args.ntesterror) + " ...")

        ffnoise = ff + torch.normal(0.0 , ffstd.real)

        fffnoise = ffnoise.to(torch.complex128)

        hwffnoise = torch.einsum('ijklmn , m , jlmn -> ijkl' , h , w , fffnoise)
        
        r1noise = getR1(args.pzero , hwffnoise , pp_tt , pw_tt , nucm , args.maxpval)

        r1noise = r1noise.reshape(
                        (
                            hwffnoise.shape[0] * hwffnoise.shape[1] , 
                            hwffnoise.shape[2] * hwffnoise.shape[3]
                        )
                    )

        r1noise = torch.eye(r1noise.shape[0]) + r1noise

        resnoise = torch.linalg.solve(r1noise , hwffnoise[: , : , args.dnum , args.mnum].reshape(hwffnoise.shape[0] * hwffnoise.shape[1]))

        resnoise = resnoise.reshape((hwffnoise.shape[0] , hwffnoise.shape[1]))

        bnoise = resnoise[args.dnum ,  -1]

        snoise = 1.0 - 1.0j * args.pzero * nucm * torch.pi * bnoise

        arg_snoise = 0.5 * torch.arctan2(snoise.imag , snoise.real) * 180 / torch.pi

        with open(os.path.join(args.dir , "arg_s_check") , "a") as f:
            f.write(str(arg_snoise.item()) + "\n")

        del ffnoise
        del hwffnoise
        del r1noise
        del resnoise

