set terminal png
set title "Radiation Coupling Test"
set xlabel "t"
set ylabel "T"
set out "temp1.png"
plot "temp1.dat" u 1:2 t "Tgas", "temp1.dat" u 1:3 t "Trad"
set out "temp2.png"
plot "temp2.dat" u 1:2 t "Tgas", "temp2.dat" u 1:3 t "Trad"

