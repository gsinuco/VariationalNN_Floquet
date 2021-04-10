reset
unset logscale
set term x11 enhanced 0
set size 2,1
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
#set xrange [-0.15:0.15]
unset key
rgbfudge(x) = x*32760 + (2.0*pi-x)*1280 + int(abs(pi-x)*100/9.)
set terminal postscript eps color "Helvetica,14"
set output "FloquetvsRBM_Omega.eps"
#set colorbox user
#set colorbox horizontal origin 0,0,0#screen 0.05, 0.05 size screen 0.9, 0.05 front  noinvert bdefault
#show colorbox
set multiplot #layout 1, 3 title "Multiplot layout 1, 3" font ",14"
unset label 
set size 1,1.3
set origin 0,0
set label "(a)" at 0,-26,0.09 front font ",30"
splot "absPsi_comparison_10th_3D.dat" i 60  u ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):4:(rgbfudge(abs($5)))  w boxes fc rgb variable
set size 1,1.3
set origin 1,0
unset label
set label "(b)" at 0,-27,0.09 front font ",30"
set grid
splot "absPsi_comparison_10th_3D.dat" i 60 u  ($2<0?$2+0.4:$2-0.4):($3>=-9 & $3<=9?$3:1/0):8:(rgbfudge(abs($9)))  w boxes fc rgb variable
#set size 1,0.85
#set origin 2,0.1
#set autoscale xy
#set logscale y
#unset key
#unset grid
#set xlabel "Iteration"
#set ylabel "{/Symbol G}"
#set label "(c)" at -100,7 font ",30"
#set xrange [0:1024]
#set yrange [1e-6:3]
#set xtics 0,100 offset 0
#set ytics 1e-6,10 
#set format y '10^{%T}'
#plot "absPsi_comparison_Floquet.dat" i 1 u 1 w l  lt 3
unset multiplot
unset output


reset
set term x11 1
set autoscale xy
plot "absPsi_comparison_10th_3D.dat" i 19 u 1:3 w lp pt 5 lc rgbcolor "black", "" i 61 u 1:3 w lp pt 6 lc rgbcolor "red"