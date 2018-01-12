#!/usr/bin/gnuplot -persist

# call with:
# gnuplot -e "filename='foo.data'" -e "outfile='bar.png'" scriptname.plt
set title "Computation Time with different Kernels"
set pointintervalbox 3
set terminal png size 1920,1200 enhanced font "Helvetica,20"
set output outfile
set key bottom left
set xrange [1:]
set xtics 1
set yrange [5:]
#set terminal postscript eps enhanced color font 'Helvetica,10'
set style line 1 lc rgb '#BB1100' lt 1 lw 2 pt 7 ps 1.5   # --- blueset xlabel "X"
set style line 2 lc rgb '#AA2200' lt 1 lw 1 pt 7 ps 1.5   # --- blueset xlabel "X"
set style line 3 lc rgb '#884400' lt 1 lw 1 pt 7 ps 1.5   # --- blueset xlabel "X"
set style line 4 lc rgb '#446600' lt 1 lw 1 pt 7 ps 1.5   # --- blueset xlabel "X"
set style line 5 lc rgb '#006633' lt 1 lw 1 pt 7 ps 1.5   # --- blueset xlabel "X"
set style line 6 lc rgb '#006699' lt 1 lw 2 pt 7 ps 1.5   # --- blueset xlabel "X"
set style line 7 lc rgb '#996699' lt 1 lw 2 pt 7 ps 1.5   # --- blueset xlabel "X"
        
plot filename using 1:2 title 'Only old-style interaction' with linespoints ls 1, filename using 1:3 title 'With new multipole multipole kernel' with linespoints ls 2,filename using 1:4 title 'With new multipole monopole kernel' with linespoints ls 3,filename using 1:5 title 'With new monopole monopole kernel' with linespoints ls 4,filename using 1:6 title 'With new monopole multipole kernel' with linespoints ls 5,filename using 1:7 title 'With all new kernels at once' with linespoints ls 6,filename using 1:8 title 'With all new kernels except p2m' with linespoints ls 7


