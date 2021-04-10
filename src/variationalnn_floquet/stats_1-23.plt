
reset 
#set term postscript eps color enhanced "Helvetica,24"
#set output "EnergyError.eps"
unset key
unset label
set xlabel "{/Symbol W}"
set ylabel "{/Symbol D}E"
set label "(a)" at 0.5,45
set xrange [0:10]
set yrange [-2:16]
set ytics -20,5
plot "stats_1-10_gnuplot.dat" i 0 u 1:2:(sqrt($3)) w yerrorbars lw 3, "" i 0 u 1:2 w lp lw 3 lt 3
unset output

reset
set term wxt 11
#set term postscript eps color enhanced "Helvetica,24"
#set output "LossInitFinal.eps"
unset key
unset label
set xlabel "{/Symbol W}"
set ylabel "{/Symbol G}"
set label "(c)" at 0.5,370
#set xrange [0:10]
#set yrange [-20:50]
set ytics -50,50
plot "stats_1-10_gnuplot.dat" i 1 u 1:2:(sqrt($3)) w yerrorbars lw 3, "" i 1 u 1:2 w lp lw 3 lt 3,"stats_1-10_gnuplot.dat" i 2 u 1:2:(sqrt($3)) w yerrorbars lt 4 lw 3, "" i 2 u 1:2 w lp lw 3 lt 7 
unset output


reset
set term wxt 11
#set term postscript eps color enhanced "Helvetica,24"
#set output "Fidelity.eps"
unset key
unset label
set xlabel "{/Symbol W}"
set ylabel "Fidelity" offset 2,0
set label "(b)" at 0.5,1.1 front
set yrange [-0.4:1.25]
set ytics 0,0.5
plot "stats_1-10_gnuplot.dat" i 3 u 1:2:(sqrt($3)) w yerrorbars lw 3, "" i 3 u 1:2 w lp lw 3 lt 3
unset output


reset
set term wxt 1
unset key
set xlabel "{/Symbol W}"
set ylabel "Qubit Gap"
set logscale y 
plot "QubitGapSamples_gnuplot.dat" u 1:2 w lp lw 3,  "" u 1:4 w lp lw 3, "" u 1:5 w lp lw 1

