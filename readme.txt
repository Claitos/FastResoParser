
- Format 1 (e.g., PDG2005.dat) 
  Both particles and anti-particles are included. The photon is
  always included. There are 12 columns, containing:
	
  1.    ID 
  2.    Name 
  3.    Mass(GeV) 
  4.    Width(GeV) 
  5.    Degeneracy(from spin) 
  6.    Baryon no. 
  7.    Strangeness no. 
  8.    Charm no. 
  9.    Bottom no.
  10.   Isospin
  11.   Electric charge
  12.   No. of decay channels

  Example: Delta(1232)++
  ID   Name          M(GeV) W(GeV) Deg B(aryon) S C B(ottom) I   Q  Dec.No
  -------------------------------------------------------------------------
  2224 Delta(1232)++ 1.232  0.12   4   1        0 0 0        3/2 2  1
 

------------------------------------------------------------------------------

- Format 1 (e.g., decays_PDG2005.dat) 
  Both particles and anti-particles are included. The photon is
  always included. Each particle appears in the following way. A 12-column
  line repeating the entry from the particle list, followed by a number of
  8-column lines, each representing one decay mode.

	The number of 8-column lines is the same as the 12th entry in the 12-column
  line.
	
  Each 8-column line is structured as follows:
  1.    ID (mother particle)
  2.    No. of daughter particles
  3.    Branching ratio
  4.-8. ID (daughter particle, 0 if none)

  Example: Delta(1232)++
  ID   Name          M(GeV) W(GeV) Deg B(aryon) S C B(ottom) I   Q  Dec.No
  -------------------------------------------------------------------------
  2224 Delta(1232)++ 1.232  0.12   4   1        0 0 0        3/2 2  1

  ID (mother)  No. daughters  BR  ID#1  ID#2  ID#3  ID#4  ID#5
  ------------------------------------------------------------------------
  2224	       2	            1	  2212	211	  0	    0	    0				
