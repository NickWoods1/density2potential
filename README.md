*****************
density2potential
*****************

`density2potental` is a python package designed to generate a Kohn-Sham
effective potential that generated a reference density.

*************
Functionality
*************

`exact` computes an exact time-dependent density from a given 
time-dependent external potential, where the particles
interact via the softened Coulomb potential. Implemented
for 1- and 2-particle systems. 

`find-vks` obtains the Kohn-Sham effective potential that 
computes generates a given reference density. This reference
density can be fed in from `exact`, or elsewhere. 
