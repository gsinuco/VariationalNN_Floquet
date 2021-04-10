reset 
index_ = 12

set yrange [0:0.41]
set xrange [18:45]

set ytics 0,0.1

set term x11 0
unset label 
unset key
set label "(b) RBM reconstruction" at 20,0.37
set xlabel "Channel index"
set ylabel "|<|>|"
#set xrange [0:66]
plot "absPsi_comparison_10th.dat" i index_ u ($2**2) w lp lw 3, "absPsi_comparison_10th.dat" i index_ u ($1**2) w lp lt 3 lw 3

set term x11 1
unset label 
set label "(a) Floquet" at 20,0.37
plot "absPsi_comparison_10th.dat" i index_ u ($3**2) w lp lw 3, "absPsi_comparison_10th.dat" i index_ u ($4**2) w lp lt 3 lw 3


index_ = 31

set term x11 2
unset label 
unset key
set yrange [0:0.32]
set label "(d) RBM reconstruction" at 20,0.28
set xlabel "Channel index"
set ylabel "|<|>|"
#set xrange [0:66]
plot "absPsi_comparison_10th.dat" i index_ u ($2**2) w lp lw 3, "absPsi_comparison_10th.dat" i index_ u ($1**2) w lp lt 3 lw 3

set term x11 3
unset label 
set yrange [0:0.22]
set label "(c) Floquet" at 20,0.19
plot "absPsi_comparison_10th.dat" i index_ u ($3**2) w lp lw 3, "absPsi_comparison_10th.dat" i index_ u ($4**2) w lp lt 3 lw 3



reset
set term x11 4
plot "absPsi_comparison_Floquet.dat" i 0 u 1 w lp, "absPsi_comparison_Floquet.dat" i 0 u 2 w lp
set term x11 5
plot "absPsi_comparison_Floquet.dat" i 0 u 3 w lp, "absPsi_comparison_Floquet.dat" i 0 u 4 w lp

reset 
plot "absPsi_comparison_Floquet.dat" i 1 u 1 w lp

reset
set term x11 6
plot "absPsi_comparison_Floquet.dat" i 2 u 1 w lp, "absPsi_comparison_Floquet.dat" i 2 u 2 w lp
set term x11 7
plot "absPsi_comparison_Floquet.dat" i 2 u 3 w lp, "absPsi_comparison_Floquet.dat" i 2 u 4 w lp


reset
set term x11 6
plot "absPsi_comparison_Floquet.dat" i 4 u 1 w lp, "absPsi_comparison_Floquet.dat" i 2 u 2 w lp
set term x11 7
plot "absPsi_comparison_Floquet.dat" i 4 u 3 w lp, "absPsi_comparison_Floquet.dat" i 2 u 4 w lp

reset 
set logscale y
unset key
set xlabel "Iteration"
set ylabel "{/Symbol G}"
set xrange [1:1024]
plot "absPsi_comparison_Floquet.dat" i 1 u 1 w l lw 3 lt -1, "absPsi_comparison_Floquet.dat" i 3 u ($1)  w l lw 3 lt 0 , "absPsi_comparison_Floquet.dat" i 5 u ($1) w l lw 3 lt 3



reset
set yrange [-pi/2:pi/2]
set term x11 0
plot "absPsi_comparison_Floquet_3D.dat" i 4 u 1:4 w boxes, "" i 4 u 1:($5-0.1380) w boxes
#set term x11 1
#plot "absPsi_comparison_Floquet_3D.dat" i 4 u 1:8 w boxes, "" i 4 u 1:(($9)) w boxes
set term x11 2
plot "absPsi_comparison_Floquet_3D.dat" i 4 u 1:8 w boxes, "" i 4 u 1:(($9-0.6935)) w boxes


reset 
set term x11 3
set size 1,1.1
set xyplane at 0
set style fill   solid 1.00 border
set boxdepth 0.5
set boxwidth 0.03# absolute
ti(col) = sprintf("%d",col)
set view 44,135
#set view 90,90
set xlabel "x"
set ytics -9,1,9 offset -0.5
set grid lt -1 lc "grey"
set yrange [-10.1:10.1]
set xtics ("UP" -0.1, "DOWN" 0.1) offset 3
set ztics 0,0.2
set xlabel "Bare State"
set ylabel "Photon number" offset -3
set xrange [-0.15:0.15]
unset key
rgbfudge(x) = x*32760 + (2.0*pi-x)*1280 + int(abs(pi-x)*100/9.)
#rgbfudge(x) = x*51*32768 + (pi-x)*51*128 + int(abs(pi/2-x)*510/9.)
set terminal postscript eps color "Helvetica,14"
set output "FloquetvsRBM_Floquet3rd.eps"
#set colorbox user
#set colorbox horizontal origin 0,0,0#screen 0.05, 0.05 size screen 0.9, 0.05 front  noinvert bdefault
#show colorbox
splot "absPsi_comparison_Floquet_3D.dat" i 4 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):8:(rgbfudge(abs($9-0.6935+0.001757))) w boxes fc rgb variable
set output "FloquetvsRBM_RBM3rd.eps"
#set term x11 4
#set colorb horizontal #user origin .05, .05 size .9, .05
splot "absPsi_comparison_Floquet_3D.dat" i 4 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):4:(rgbfudge(abs($5-0.1380-0.009786))) w boxes fc rgb variable
unset output


#reset
#set yrange [-pi/2:pi/2]
#set term x11 0
#plot "absPsi_comparison_Floquet_3D.dat" i 4 u 1:4 w boxes, "" i 4 u 1:($5-0.1380-0.009786) w boxes
#set term x11 1
#plot "absPsi_comparison_Floquet_3D.dat" i 4 u 1:8 w boxes, "" i 4 u 1:(($9)) w boxes
#set term x11 2
#plot "absPsi_comparison_Floquet_3D.dat" i 4 u 1:8 w boxes, "" i 4 u 1:(($9-0.6935+0.001757)) w boxes



set term x11 enhanced 0
set size 3,1
set xyplane at 0
set style fill   solid 1.00 border
set boxdepth 0.5
set boxwidth 0.03# absolute
ti(col) = sprintf("%d",col)
set view 44,135
#set view 90,90
set xlabel "x"
set ytics -9,1,9 offset -0.5
set grid lt -1 lc "grey"
set yrange [-10.1:10.1]
set xtics ("UP" -0.1, "DOWN" 0.1) offset 3
set ztics 0,0.2
set xlabel "Bare State"
set ylabel "Photon number" offset -3
set xrange [-0.15:0.15]
unset key
rgbfudge(x) = x*32760 + (2.0*pi-x)*1280 + int(abs(pi-x)*100/9.)
set terminal postscript eps color "Helvetica,14"
set output "FloquetvsRBM_Floquet3rd.eps"
#set colorbox user
#set colorbox horizontal origin 0,0,0#screen 0.05, 0.05 size screen 0.9, 0.05 front  noinvert bdefault
#show colorbox
set multiplot #layout 1, 3 title "Multiplot layout 1, 3" font ",14"
unset label 
set size 1,1.3
set origin 0,0
set label "(g)" at 0,-26,0.09 front font ",30"
splot "absPsi_comparison_Floquet_3D.dat" i 4 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):8:(rgbfudge(abs($9-0.6935+0.001757))) w boxes fc rgb variable
set size 1,1.3
set origin 1,0
unset label
set label "(h)" at 0,-27,0.09 front font ",30"
set grid
splot "absPsi_comparison_Floquet_3D.dat" i 4 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):4:(rgbfudge(abs($5-0.1380-0.009786))) w boxes fc rgb variable
set size 1,0.85
set origin 2,0.1
set autoscale xy
set logscale y
unset key
unset grid
set xlabel "Iteration"
set ylabel "{/Symbol G}"
set label "(i)" at -100,7 font ",30"
set xrange [0:1024]
set yrange [1e-6:3]
set xtics 0,100
set ytics 1e-6,10 
set format y '10^{%T}'
plot "absPsi_comparison_Floquet.dat" i 5 u 1 w l  lt 3#, "absPsi_comparison_Floquet.dat" i 3 u ($1)  w l lw 3 lt 0 , "absPsi_comparison_Floquet.dat" i 5 u ($1) w l lw 3 lt 3
#unset output
unset multiplot


reset
unset logscale
set term x11 enhanced 0
set size 3,1
set xyplane at 0
set style fill   solid 1.00 border
set boxdepth 0.5
set boxwidth 0.03# absolute
ti(col) = sprintf("%d",col)
set view 44,135
#set view 90,90
set xlabel "x"
set ytics -9,1,9 offset -0.5
set grid lt -1 lc "grey"
set yrange [-10.1:10.1]
set xtics ("UP" -0.1, "DOWN" 0.1) offset 3
set ztics 0,0.2
set xlabel "Bare State"
set ylabel "Photon number" offset -3
set xrange [-0.15:0.15]
unset key
rgbfudge(x) = x*32760 + (2.0*pi-x)*1280 + int(abs(pi-x)*100/9.)
set terminal postscript eps color "Helvetica,14"
set output "FloquetvsRBM_Floquet2nd.eps"
#set colorbox user
#set colorbox horizontal origin 0,0,0#screen 0.05, 0.05 size screen 0.9, 0.05 front  noinvert bdefault
#show colorbox
set multiplot #layout 1, 3 title "Multiplot layout 1, 3" font ",14"
unset label 
set size 1,1.3
set origin 0,0
set label "(d)" at 0,-26,0.09 front font ",30"
splot "absPsi_comparison_Floquet_3D.dat" i 2 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):8:(rgbfudge(abs($9-0.6935+0.001757))) w boxes fc rgb variable
set size 1,1.3
set origin 1,0
unset label
set label "(e)" at 0,-27,0.09 front font ",30"
set grid
splot "absPsi_comparison_Floquet_3D.dat" i 2 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):4:(rgbfudge(abs($5-0.1380-0.009786))) w boxes fc rgb variable
set size 1,0.85
set origin 2,0.1
set autoscale xy
set logscale y
unset key
unset grid
set xlabel "Iteration"
set ylabel "{/Symbol G}"
set label "(f)" at -100,7 font ",30"
set xrange [0:1024]
set yrange [1e-6:3]
set xtics 0,100
set ytics 1e-6,10 
set format y '10^{%T}'
plot "absPsi_comparison_Floquet.dat" i 3 u 1 w l  lt 3
#unset output
unset multiplot


reset
unset logscale
set term x11 enhanced 0
set size 3,1
set xyplane at 0
set style fill   solid 1.00 border
set boxdepth 0.5
set boxwidth 0.03# absolute
ti(col) = sprintf("%d",col)
set view 44,135
#set view 90,90
set xlabel "x"
set ytics -9,1,9 offset -0.5
set grid lt -1 lc "grey"
set yrange [-10.1:10.1]
set xtics ("UP" -0.1, "DOWN" 0.1) offset 3
set ztics 0,0.2
set xlabel "Bare State"
set ylabel "Photon number" offset -3
set xrange [-0.15:0.15]
unset key
rgbfudge(x) = x*32760 + (2.0*pi-x)*1280 + int(abs(pi-x)*100/9.)
set terminal postscript eps color "Helvetica,14"
set output "FloquetvsRBM_Floquet1st.eps"
#set colorbox user
#set colorbox horizontal origin 0,0,0#screen 0.05, 0.05 size screen 0.9, 0.05 front  noinvert bdefault
#show colorbox
set multiplot #layout 1, 3 title "Multiplot layout 1, 3" font ",14"
unset label 
set size 1,1.3
set origin 0,0
set label "(a)" at 0,-26,0.09 front font ",30"
splot "absPsi_comparison_Floquet_3D.dat" i 0 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):8:(rgbfudge(abs($9-0.6935+0.001757))) w boxes fc rgb variable
set size 1,1.3
set origin 1,0
unset label
set label "(b)" at 0,-27,0.09 front font ",30"
set grid
splot "absPsi_comparison_Floquet_3D.dat" i 0 u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):4:(rgbfudge(abs($5-0.1380-0.009786))) w boxes fc rgb variable
set size 1,0.85
set origin 2,0.1
set autoscale xy
set logscale y
unset key
unset grid
set xlabel "Iteration"
set ylabel "{/Symbol G}"
set label "(c)" at -100,7 font ",30"
set xrange [0:1024]
set yrange [1e-6:3]
set xtics 0,100 offset 0
set ytics 1e-6,10 
set format y '10^{%T}'
plot "absPsi_comparison_Floquet.dat" i 1 u 1 w l  lt 3
#unset output
unset multiplot


