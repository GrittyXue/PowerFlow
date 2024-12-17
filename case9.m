function [Bus, Line,Transform] = case9
%        bindex       bmagn      bangle    power_real       power_imag         load_real     load_imag     node_type     Ixb_flag       load_type             Pe    Qe   Im   Io    
%          1           2           3          4                 5                  6             7             8          9               10                11    12   13   14 
Bus=[
           1        1.04         0           0.7164           9.99                 0             0           1           0                                  0     0    0   0
           2       1.025         0           1.63             9.99                 0             0           3           0                                  0     0    0   0
           3       1.025         0           0.85             9.99                 0              0           3           0                                  0     0    0   0
           4         1           0           0                0                    0             0           2           0                                  0     0    0   0
           5         1           0           0                0                    2             1.5         2           0                                  0     0    0   0
           6         1           0           0                0                    0.9           0.3         2           0                                  0     0    0   0
           7         1           0           0                0                    0             0           2           0                                  0     0    0   0
           8         1           0           0                0                    1             0.35        2           0                                  0     0    0   0
           9         1           0           0                0                    0             0           2           0                                  0     0    0   0
];
%          i_bus      j_bus       lr         lx           lib        lib          lk     lnum
%           1          2          3          4             5        6           7        8
%
Line=[
           4           5        0.01       0.085       0.088     0.088           1         1         
           4           6       0.017       0.092       0.079     0.079           1         1   
           5           7       0.032       0.161       0.153     0.153           1         1 
           6           9       0.039        0.17       0.179     0.179           1         1 
           7           8      0.0085       0.072      0.0745    0.0745           1         1 
           8           9      0.0119      0.1008      0.1045    0.1045           1         1 
];

%          i_bus      j_bus       lr         lx           lib        lib          lk     lnum
%           1           2         3         4              5          6            7
                                        
Transform=[
          1           4           0.01      0.0576           0           0           1
           2           7           0.01      0.0625           0           0           1
           3           9           0.01      0.0586           0           0           1
];
return;