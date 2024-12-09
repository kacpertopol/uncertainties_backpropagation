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
    parser.add_argument("mnum" , help = "momentum number for initial state" , type = int)
    parser.add_argument("--dnums" , help = "discrete numbers for initial state" , action = "append")
    parser.add_argument("--maxpval" , help = "Maximum momentum in fm^-1." , type = float , default = 10.0)
    parser.add_argument("--ppts" , help = "Number of points for momenta." , type = int , default = 64)
    parser.add_argument("--xpts" , help = "Number of points for x." , type = int , default = 64)
    parser.add_argument("--relerror" , help = "Relative error for scalar functions" , type = str , default = "0 0 0 0 0 0")
    parser.add_argument("--ntesterror" , help = "Number of samples to test error" , type = int)
    parser.add_argument("--forcepath" , help = "Path to program calculating scalar functions" , default = "./ff")
    args = parser.parse_args()
 
    dnums = list(map(lambda x : tuple(map(int , x.split())) , args.dnums))

    if(os.path.isdir(args.dir)):
        sys.stderr.write("Directory " + args.dir + " exists. Exiting.\n")
        sys.exit(1)

    os.mkdir(args.dir)

    with open(os.path.join(args.dir , "cmd") , "w") as f:
        f.write(" ".join(sys.argv))
    shutil.copyfile(os.path.realpath(__file__) , os.path.join(args.dir , "test_scattering_coupled.py"))


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

    # nucleon mass fm^-1
    nucm = 0.5 * (938.272 + 939.565) / 197.328

    # torch.tensor momentum points and weights 
    pp_tt = pwData.getTorchPointsP()
    pw_tt = pwData.getTorchWeightsP()
    log(args.dir , "saving torch tensor points and weights for momenta: pp_tt.pt, pw_tt.pt ...")
    torch.save(pp_tt , os.path.join(args.dir , "pp_tt.pt"))
    torch.save(pw_tt , os.path.join(args.dir , "pw_tt.pt"))

    r1 = getR1(args.pzero , hwff , pp_tt , pw_tt , nucm , args.maxpval)

    r1 = r1.reshape(
                    (
                        hwff.shape[0] * hwff.shape[1] , 
                        hwff.shape[2] * hwff.shape[3]
                    )
                )

    r1 = torch.eye(r1.shape[0]) + r1

    res0 = torch.linalg.solve(r1 , hwff[: , : , 0 , args.mnum].reshape(hwff.shape[0] * hwff.shape[1]))

    res0 = res0.reshape((hwff.shape[0] , hwff.shape[1]))

    res1 = torch.linalg.solve(r1 , hwff[: , : , 1 , args.mnum].reshape(hwff.shape[0] * hwff.shape[1]))

    res1 = res1.reshape((hwff.shape[0] , hwff.shape[1]))

    log(args.dir , "saving torch tensor result to res0.pt ...")
    torch.save(res0 , os.path.join(args.dir , "res0.pt"))
    
    log(args.dir , "saving torch tensor result to res1.pt ...")
    torch.save(res1 , os.path.join(args.dir , "res1.pt"))
    
    log(args.dir , "saving result to res0re, res0im ...")
    with open(os.path.join(args.dir , "res0re") , "w") as f:
        for i in range(hwff.shape[0]):
            for j in range(hwff.shape[1]):
                f.write(str(res0[i , j].real.item()) + " ")
            f.write("\n")
    with open(os.path.join(args.dir , "res0im") , "w") as f:
        for i in range(hwff.shape[0]):
            for j in range(hwff.shape[1]):
                f.write(str(res0[i , j].imag.item()) + " ")
            f.write("\n")
    
    log(args.dir , "saving result to res1re, res1im ...")
    with open(os.path.join(args.dir , "res1re") , "w") as f:
        for i in range(hwff.shape[0]):
            for j in range(hwff.shape[1]):
                f.write(str(res1[i , j].real.item()) + " ")
            f.write("\n")
    with open(os.path.join(args.dir , "res1im") , "w") as f:
        for i in range(hwff.shape[0]):
            for j in range(hwff.shape[1]):
                f.write(str(res1[i , j].imag.item()) + " ")
            f.write("\n")

    log(args.dir , "saving momenta to pppp ...")
    with open(os.path.join(args.dir , "pppp") , "w") as f:
        for i in range(len(pp)):
            f.write(str(i) + " " + str(pp[i]) + " " + str(pw[i]) + "\n")

    log(args.dir , "saving discrete numbers to num ...")
    with open(os.path.join(args.dir , "num") , "w") as f:
        for i in range(len(pwData.nums)):
            f.write(str(i) + " " + str(pwData.nums[i]) + "\n")

    b00 = res0[0 ,  -1]
    b01 = res0[1 ,  -1]
    b10 = res1[0 ,  -1]
    b11 = res1[1 ,  -1]

    s00 = 1.0 - 1j * args.pzero * nucm * torch.pi * b00
    s01 =     - 1j * args.pzero * nucm * torch.pi * b01
    s10 =     - 1j * args.pzero * nucm * torch.pi * b10
    s11 = 1.0 - 1j * args.pzero * nucm * torch.pi * b11
   
    deltam = 0.5 * torch.arctan2(s00.imag , s00.real)
    deltap = 0.5 * torch.arctan2(s11.imag , s11.real)

    eps = 0.5 * torch.asin((-1j * s01 * torch.exp(-1j * (deltam + deltap))).real)

    deltamdeg = deltam * 180 / torch.pi
    deltapdeg = deltap * 180 / torch.pi
    epsdeg = eps * 180 / torch.pi

    log(args.dir , "got DELTEM in DEG : " + str(deltamdeg.item()))
    log(args.dir , "got DELTAP in DEG : " + str(deltapdeg.item()))
    log(args.dir , "got EPS in DEG : " + str(epsdeg.item()))

    deltamdeg.backward(retain_graph = True)

    log(args.dir , "saving gradient of DELTAM in DEG to deltamgrad.pt")
    torch.save(ff.grad , os.path.join(args.dir , "deltamgrad.pt"))

    var = torch.sum((ffstd * ff.grad) * (ffstd * ff.grad))
    varsqrt = torch.sqrt(var)

    log(args.dir , "got error for DELTAM in DEG : " + str(varsqrt.item()))

    ff.grad.data.zero_() 
 
    deltapdeg.backward(retain_graph = True)

    log(args.dir , "saving gradient of DELTAP in DEG to deltamgrad.pt")
    torch.save(ff.grad , os.path.join(args.dir , "deltamgrad.pt"))

    var = torch.sum((ffstd * ff.grad) * (ffstd * ff.grad))
    varsqrt = torch.sqrt(var)

    log(args.dir , "got error for DELTAP in DEG : " + str(varsqrt.item()))

    ff.grad.data.zero_()

    epsdeg.backward(retain_graph = True)

    log(args.dir , "saving gradient of EPS in DEG to deltamgrad.pt")
    torch.save(ff.grad , os.path.join(args.dir , "deltamgrad.pt"))

    var = torch.sum((ffstd * ff.grad) * (ffstd * ff.grad))
    varsqrt = torch.sqrt(var)

    log(args.dir , "got error for EPS in DEG : " + str(varsqrt.item()))

    ff.grad.data.zero_() 

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

        res0noise = torch.linalg.solve(r1noise , hwffnoise[: , : , 0 , args.mnum].reshape(hwffnoise.shape[0] * hwffnoise.shape[1]))

        res0noise = res0noise.reshape((hwffnoise.shape[0] , hwffnoise.shape[1]))

        res1noise = torch.linalg.solve(r1noise , hwffnoise[: , : , 1 , args.mnum].reshape(hwffnoise.shape[0] * hwffnoise.shape[1]))

        res1noise = res1noise.reshape((hwffnoise.shape[0] , hwffnoise.shape[1]))

        b00noise = res0noise[0 ,  -1]
        b01noise = res0noise[1 ,  -1]
        b10noise = res1noise[0 ,  -1]
        b11noise = res1noise[1 ,  -1]

        s00noise = 1.0 - 1j * args.pzero * nucm * torch.pi * b00noise
        s01noise =     - 1j * args.pzero * nucm * torch.pi * b01noise
        s10noise =     - 1j * args.pzero * nucm * torch.pi * b10noise
        s11noise = 1.0 - 1j * args.pzero * nucm * torch.pi * b11noise
       
        deltamnoise = 0.5 * torch.arctan2(s00noise.imag , s00noise.real)
        deltapnoise = 0.5 * torch.arctan2(s11noise.imag , s11noise.real)

        epsnoise = 0.5 * torch.asin((-1j * s01noise * torch.exp(-1j * (deltamnoise + deltapnoise))).real)

        deltamdegnoise = deltamnoise * 180 / torch.pi
        deltapdegnoise = deltapnoise * 180 / torch.pi
        epsdegnoise = epsnoise * 180 / torch.pi

        with open(os.path.join(args.dir , "all_check") , "a") as f:
            f.write(str(deltamdegnoise.item()) + " " + str(deltapdegnoise.item()) + " " + str(epsdegnoise.item()) + "\n")

        del ffnoise
        del hwffnoise
        del r1noise
        del res0noise
        del res1noise

