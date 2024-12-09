#!/usr/bin/env python3

import numpy as np
import partial
import torch
import matplotlib.pyplot as plt

import os
import sys
import argparse

import shutil
import glob
from pathlib import Path

def log(dir , message):
    with open(os.path.join(dir , "log") , 'a') as f:
        f.write(message + "\n")

# returns the matrix representation of
# (E - H0)^-1 V
# e - is the energy
# hwff - array containing the pw dec of the potential 
# pp , pw - are the points and weights for momenta
# m - is the nucleon mass
def getR(e , hwff , pp , pw , m):
    wpp = pw * pp * pp
    vv = hwff * wpp[None , None , None , :]
    eh0 = 1.0 / (e - ((pp * pp) / m))
    return (eh0[None , : , None , None] * vv).reshape(
                    (
                        hwff.shape[0] * hwff.shape[1] , 
                        hwff.shape[2] * hwff.shape[3]
                    )
                )

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description = "Deuteron calculation with error propagation.")
    parser.add_argument("dir" , help = "Output directory.")
    parser.add_argument("--maxpval" , help = "Maximum momentum in fm^-1." , type = float , default = 10.0)
    parser.add_argument("--ppts" , help = "Number of points for momenta." , type = int , default = 64)
    parser.add_argument("--xpts" , help = "Number of points for x." , type = int , default = 64)
    #parser.add_argument("--action" , "-a" , action = "append")
    parser.add_argument("--dpts" , help = "Number of points for deuteron loop." , type = int , default = 401)
    parser.add_argument("--dctr" , help = "Deuteron energy ceter for loop fm^-1." , type = float , default = -2.22099 / 197.328)
    parser.add_argument("--dd" , help = "Deuteron energy change for loop fm^-1." , type = float , default = 0.01)
    parser.add_argument("--relerror" , help = "Relative error for scalar functions" , type = str , default = "0 0 0 0 0 0")
    parser.add_argument("--testerror" , help = "Number of iteration to test error." , type = int , action = "append")
    parser.add_argument("--ntesterror" , help = "Number of samples to test error" , type = int)
    args = parser.parse_args()
  
    if(os.path.isdir(args.dir)):
        sys.stderr.write("Directory " + args.dir + " exists. Exiting.\n")
        sys.exit(1)

    os.mkdir(args.dir)

    with open(os.path.join(args.dir , "cmd") , "w") as f:
        f.write(" ".join(sys.argv))
    shutil.copyfile(os.path.realpath(__file__) , os.path.join(args.dir , "test_deuteron.py"))


    # relative errors
    rer = np.array(list(map(lambda x : float(x) , args.relerror.split())))

    # points and weights
    log(args.dir , "getting points and weights ...")

    pp , pw = np.polynomial.legendre.leggauss(args.ppts)
    pp = (pp + 1.0) * 0.5 * args.maxpval
    pw = pw * 0.5 * args.maxpval
    xp , xw = np.polynomial.legendre.leggauss(args.xpts)

    log(args.dir , "saving points and weights for momenta and x to: pp.npy, pw.npy, xp.npy , xw.npy ...")
    np.save(os.path.join(args.dir , "pp.npy") , pp)
    np.save(os.path.join(args.dir , "pw.npy") , pw)
    np.save(os.path.join(args.dir , "xp.npy") , xp)
    np.save(os.path.join(args.dir , "xw.npy") , xw)

    # pwd data
    log(args.dir , "calculating pw data ...")

    pwData = partial.PWD(pp , pw , xp , xw , restrict = None)

    # integrands from equation (7) from [Eur. Phys. J. A 43 241-250 (2010)] 
    h = pwData.getTorchTensor()

    # torch.tensor weights for x integration
    w = pwData.getTorchWeightsX()
    
    # values of the scalar functions
    log(args.dir , "getting scalar function values ...")
    
    ff = pwData.getTorchInput("./ff")
    ff.requires_grad = True

    log(args.dir , "saving discrete numbers to num ...")
    with open(os.path.join(args.dir , "num") , "w") as f:
        for i in range(len(pwData.nums)):
            f.write(str(i) + " " + str(pwData.nums[i]) + "\n")
    
    # calculating errors
    ffstd = ff.abs() * torch.from_numpy(rer[None , None , None , :])

    # i - discrete number of the partal wave in the final state
    # j - numbers momentum in the final state
    # k - discrete number of the partial wave in the inital state
    # l - numbers momentum in the initial state
    log(args.dir , "calculating pw decomposition ...")

    hwff = torch.einsum('ijklmn , m , jlmn -> ijkl' , h , w , ff)

    # nucleon mass fm^-1
    nucm = 0.5 * (938.272 + 939.565) / 197.328

    # torch.tensor momentum points and weights 
    pp_tt = pwData.getTorchPointsP()
    pw_tt = pwData.getTorchWeightsP()
    log(args.dir , "saving torch tensor points and weights for momenta: pp_tt.pt, pw_tt.pt ...")
    torch.save(pp_tt , os.path.join(args.dir , "pp_tt.pt"))
    torch.save(pw_tt , os.path.join(args.dir , "pw_tt.pt"))


    # loop
    log(args.dir , "starting loop ...")
    ed = args.dctr
    mn = None
    mne = None
    te = []
    tv = []
    tverr = []
    ii = 0
    ls = torch.linspace(-args.dd , args.dd , args.dpts)
    for dd in ls:
        log(args.dir , "iteration of main loop: " + str(ii) + " of " + str(ls.shape[0]) + "...")
        log(args.dir , "dd : " + str(dd.item()) + " , args.dd : " + str(args.dd))
        if(ff.grad is not None):
            ff.grad = None
        e = ed + dd
        r = getR(e , hwff , pp_tt , pw_tt , nucm)
        evr = torch.linalg.eigvals(r)
        evrs , indexes = torch.sort(torch.abs(evr - 1.0))
        if((mn is None) or (evrs[0].item() < mn)):
            mn = evrs[0].item()
            mne = e.item()
        te.append(e.item())
        tv.append(evr[indexes[0]].item())
        evrs[0].real.backward(retain_graph = True)
        var = torch.sum((ffstd * ff.grad) * (ffstd * ff.grad))
        tverr.append(torch.sqrt(var).real.item())
        log(args.dir , "for energy MeV : " + str(197.328 * te[-1]) + " eigenvalue: " + str(tv[-1]) + " error: " + str(tverr[-1]))
        if((args.testerror is not None) and (args.ntesterror is not None) and (ii in args.testerror)):
            log(args.dir , "checking error ...")
            tvs = []
            for iii in range(args.ntesterror):
                log(args.dir , "iteration " + str(iii) + " of " + str(args.ntesterror) + " ...")
                ffnoise = ff + torch.normal(0.0 , ffstd.real)
                hwffnoise = torch.einsum('ijklmn , m , jlmn -> ijkl' , h , w , ffnoise)
                rnoise = getR(e , hwffnoise , pp_tt , pw_tt , nucm)
                evrnoise = torch.linalg.eigvals(rnoise)
                evrsnoise , indexesnoise = torch.sort(torch.abs(evrnoise - 1.0))
                tvs.append(evrnoise[indexesnoise[0]].item())
                del ffnoise
                del hwffnoise
                del rnoise

            tvsnp = np.array(tvs)
            log(args.dir , "stdev : " + str(np.std(tvsnp.real)))
            log(args.dir , "saving eigenvalues to tvsnp" + str(ii) + ".npy ...")
            np.save(os.path.join(args.dir , "tvsnp" + str(ii) + ".npy") , tvsnp)

        ii += 1

    log(args.dir , "saving energies fm^-1 and evalues, errors from loop to: te.npy, tv.npy, tverr.npy ...")
    np.save(os.path.join(args.dir , "te.npy") , np.array(te))
    np.save(os.path.join(args.dir , "tv.npy") , np.array(tv))
    np.save(os.path.join(args.dir , "tverr.npy") , np.array(tverr))

    # calculate for energy with eigenvalue closest to 1
    log(args.dir , "calculating for energy with eigenvalue closest to 1 ...")
    ff.requires_grad = True
    
    r = getR(mne , hwff , pp_tt , pw_tt , nucm)
    log(args.dir , "saving matrix for energy " + str(mne) + " to r.pt ...")
    torch.save(r , os.path.join(args.dir , "r.pt"))
    evr = torch.linalg.eigvals(r)
    evrs , indexes = torch.sort(torch.abs(evr - 1.0))

    log(args.dir , "for energy : " + str(mne * 197.328) + " MeV , |eigenvalue - 1| :" + str(evrs[0].item()))
    log(args.dir , "eigenvalue real part : " + str(evr[indexes[0]].real.item()))
    log(args.dir , "eigenvalue imaginary part : " + str(evr[indexes[0]].imag.item()))

    log(args.dir , "calculating gradient ...")
    evr[indexes[0]].real.backward()
    log(args.dir , "saving ff to ffsave.pt ...")
    torch.save(ff , os.path.join(args.dir , "ffsave.pt"))
    log(args.dir , "saving ff.grad to ffgradsave.pt ...")
    torch.save(ff.grad , os.path.join(args.dir , "ffgradsave.pt"))

    log(args.dir , "saving eigenvector and gradient ...")

    with torch.no_grad():
        eigenvalues , eigenvectors = torch.linalg.eig(r)
        sort , ind = torch.sort(torch.abs(eigenvalues - 1.0))
        log(args.dir , "|eigenvalue - 1| : " + str(sort[:10].numpy()) + "...")
        log(args.dir , "eigenvalue : " + str(eigenvalues[ind[0]].item()))
        log(args.dir , "eigenvectors[: , ind[0]].shape : " + str(eigenvectors[: , ind[0]].shape))
        log(args.dir , "saving eigenvector to evec.pt ...")
        torch.save(eigenvectors[: , ind[0]].reshape((hwff.shape[0] , hwff.shape[1])) , os.path.join(args.dir , "evec.pt"))

    te = np.load(os.path.join(args.dir , "te.npy"))
    tv = np.load(os.path.join(args.dir , "tv.npy"))
    tverr = np.load(os.path.join(args.dir , "tverr.npy"))

    plotsDir = os.path.join(args.dir , "plts")
    
    os.mkdir(plotsDir)

    with open(os.path.join(plotsDir , "evverr") , "w") as f:
        for i in range(te.shape[0]):
            f.write(str(te[i]) + " " + str(tv[i].real) + " " + str(tverr[i]) + "\n")

    check = glob.glob(os.path.join(args.dir , "tvsnp") + "*.npy")

    for tv in check:
        
        nme = Path(tv).stem

        tvsnp = np.load(tv)

        with open(os.path.join(plotsDir , nme) , "w") as f:
            for i in range(tvsnp.shape[0]):
                f.write(str(tvsnp[i].real) + " " + str(tvsnp[i].imag) + "\n")

#if(False):
#    # points and weights for momenta
#    print("setting up points and weights ...")
#    pp , pw = np.polynomial.legendre.leggauss(64)
#    pp = (pp + 1.0) * 0.5 * 10.0
#    pw = pw * 0.5 * 10.0
#    print("   weights for p sum to: " , pw.sum())
#    print("   p min , max : " , pp.min() , pp.max())
#    # points and weights for x
#    xp , xw = np.polynomial.legendre.leggauss(64)
#
#    # pwd data
#    print("getting pwd data ...")
#    pwData = partial.PWD(pp , pw , xp , xw)
#
#    # integrands from equation (7) from [Eur. Phys. J. A 43 241-250 (2010)] 
#    h = pwData.getTorchTensor()
#    # weights for x integration
#    w = pwData.getTorchWeightsX()
#
#    # values of the scalar functions
#    print("getting scalar function values ...")
#    ff = pwData.getTorchInput("./ff")
#
#    # i - discrete number of the partal wave in the final state
#    # j - numbers momentum in the final state
#    # k - discrete number of the partial wave in the inital state
#    # l - numbers momentum in the initial state
#    print("calculating pwd ...")
#    hwff = torch.einsum('ijklmn , m , jlmn -> ijkl' , h , w , ff)
#    print("   hwff.shape : " , hwff.shape)
#    #print("   " , hwff.shape[0] * hwff.shape[1] , hwff.shape[2] * hwff.shape[3])
#
#    # nucleon mass [fm^-1]
#    nucm = 0.5 * (938.272 + 939.565) / 197.328
#    pp = pwData.getTorchPointsP()
#    pw = pwData.getTorchWeightsP()
#    torch.save(pp , "pp.pt")
#
#    # returns the matrix representation of
#    # (E - H0)^-1 V
#    # e - is the energy
#    # hwff - array containing the pw dec of the potential 
#    # pp , pw - are the points and weights for momenta
#    # m - is the nucleon mass
#    def getR(e , hwff , pp , pw , m):
#        wpp = pw * pp * pp
#        vv = hwff * wpp[None , None , None , :]
#        eh0 = 1.0 / (e - ((pp * pp) / m))
#        return (eh0[None , : , None , None] * vv).reshape(
#                        (
#                            hwff.shape[0] * hwff.shape[1] , 
#                            hwff.shape[2] * hwff.shape[3]
#                        )
#                    )
#
#    # deuteron energy in MeV
#    ed = -2.22099 / 197.328
#
#    # loop
#    print("starting loop ...")
#    mn = None
#    mne = None
#    te = []
#    tv = []
#    for dd in torch.linspace(-0.01 , 0.01 , 401):
#        e = ed + dd
#        r = getR(e , hwff , pp , pw , nucm)
#        evr = torch.linalg.eigvals(r)
#        evrs , indexes = torch.sort(torch.abs(evr - 1.0))
#        if((mn is None) or (evrs[0].item() < mn)):
#            mn = evrs[0].item()
#            mne = e.item()
#        te.append(e.item())
#        tv.append(evrs[0].item())
#
#    # plot energies
#    plt.plot([e * 197.328 for e in te] , tv)
#    plt.show()
#
#    # calculate for energy with eigenvalue closest to 1
#    ff.requires_grad = True
#
#    r = getR(mne , hwff , pp , pw , nucm)
#    torch.save(r , "r.pt")
#    evr = torch.linalg.eigvals(r)
#    evrs , indexes = torch.sort(torch.abs(evr - 1.0))
#
#    print("   for energy : " , mne * 197.328 , " MeV , eigenvalue - 1 :" , evrs[0].item())
#    print("   eigenvalue real part : " , evr[indexes[0]].real.item())
#    print("   eigenvalue imaginary part : " , evr[indexes[0]].imag.item())
#
#    # calculate gradient
#    print("calculating gradient ...")
#
#    evr[indexes[0]].real.backward()
#    torch.save(ff , "ffsave.pt")
#    torch.save(ff.grad , "ffgradsave.pt")
#
#    print("saving eigenvector and gradient ...")
#
#    with torch.no_grad():
#        eigenvalues , eigenvectors = torch.linalg.eig(r)
#        sort , ind = torch.sort(torch.abs(eigenvalues - 1.0))
#        print("   eigenvalue - 1 : " , sort[:10].numpy() , "...")
#        print("   eigenvalue : " , eigenvalues[ind[0]].item())
#        print("   eigenvectors[: , ind[0]].shape : " , eigenvectors[: , ind[0]].shape)
#        torch.save(eigenvectors[: , ind[0]] , "evec.pt")
#
#    print("discrete numbers ...")
#
#    for i in range(len(pwData.nums)):
#        print(i , pwData.nums[i])
