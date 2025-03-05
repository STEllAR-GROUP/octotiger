set title "Radiation Diffusion Convergence Test"
set xlabel "x"
set ylabel "E"
set terminal png 
set out "dif.png"
plot  "1.dat" u 1:17 w lp t "level 1", "2.dat" u 1:17 w lp t "level 2", "3.dat" u 1:17 w lp t "level 3", "4.dat" u 1:17 w lp t "level 4", (2)**(-1.5)*exp(-x*x*300.0/4.0/2) t "analytic"

