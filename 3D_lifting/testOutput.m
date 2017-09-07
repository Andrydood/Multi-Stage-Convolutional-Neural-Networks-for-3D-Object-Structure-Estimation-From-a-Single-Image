close all
clear all


x = readNPY('../x_out.npy');
y = readNPY('../y_out.npy');
z = readNPY('../z_out.npy');

printChair(x,y,z)
axis([-3 3 -3 3 -3 3])
view(2)
