This repository contains code and data for the publication:

[1] **Estimating theoretical uncertainties of the two-nucleon observables
by using backpropagation**
K. Topolnicki, R.Skibiński, J.Golak
*Smoluchowski Institute of Physics, Jagiellonian University, Profesora Stanisława Łojawiewicza 11, Kraków, PL-30348, Poland*
(appropriate url links will be added when available).

# deuteron calculations

The code for deuteron bound state energy calculations and uncertainty estimation is available
in:

- `test_deuteron.py`. 

This script uses the module `partial.py` which in turn requires the
user to compile `fs.f90` into a python readable module with [f2py](https://numpy.org/doc/stable/f2py/).

Details of the implementation are described in the publication [1]. Additional
information is available after running the scripts with the  `--help` or `-h` option.

Please note that the script requires the user to supply the `ff` programm, see **additional requirements**. 

# scattering calculations

The code for scattering calculations and uncertainty estimation is available
in:

- `test_scattering.py`
- `test_scattering_coupled.py`


Details of the implementation are described in the publication [1]. Additional
information is available after running the scripts with the  `--help` or `-h` option.

Please note that the script requires the user to supply additional
programms to calculate the scalar function values, see **additional requirements**. 

# requirements

The following additional python libraries are required:

- `numpy`
- `pytorch`
- `matplotlib`

# additional requirements

The scripts use the module `partial.py` which in turn requires the
user to compile `fs.f90` into a python readable module with [f2py](https://numpy.org/doc/stable/f2py/).

## programs to calculate scalar function values

In order for the code to run the user must supply programms that will calculate
the scalar function values $v_{i}(p' , p , x)$ that define the two nucleon potential.
These values need to be calculated for different values of the $p'$, $p$, and $x$ 
integration points, see [1]. The command line interface of these programs is the same
for all three scripts:

- `test_deuteron.py`. 
- `test_scattering.py`
- `test_scattering_coupled.py`

If the program name is `ff` then it should be possible to call it using:

```
$ ./ff pprimepoints ppoints xpoints
```

where the three arguments are paths to text files that:

- in the first row contain the number of $p'$, $p$, or $x$ points
- in the following row contain values of consecutive $p'$, $p$, or $x$ points 

Examples of these programms may be made available upon reasonable request.
