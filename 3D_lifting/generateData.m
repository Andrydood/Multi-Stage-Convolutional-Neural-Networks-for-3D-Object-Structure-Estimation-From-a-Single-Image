close all
clear all

%Load 3D points
coordFile = load('chairKeypoints.mat');
coords = coordFile.X;

%Normalise the points

x = zeros(165,10);
y = zeros(165,10);
z = zeros(165,10);

for i = 1:165
    
    xyz = coords{1,i};
    

    [x(i,:), y(i,:), z(i,:)] = normaliseXYZ(xyz(1,:),xyz(2,:),xyz(3,:));

%     x(i,:) = norm_x;
%     y(i,:) = norm_y;
%     z(i,:) = norm_z;

end

%scatter3(x(160,:),y(160,:),z(160,:))

focal_length = 1;

z_cam = 3;

x_in = zeros(165*3*361,10);
y_in = zeros(165*3*361,10);
z_out = zeros(165*3*361,10);

for i = 1:165
    
    current_x = x(i,:);
    current_y = y(i,:);
    current_z = z(i,:);
    
    f_count = 1;
    for focal_length = 0.3:0.3:0.9
    
            for j = 0:360
                
                
                %Rotate model
                rot_x = current_z*sind(j) + current_x*cosd(j);
                rot_y = current_y;
                rot_z = current_z*cosd(j) - current_x*sind(j);
                
                %Distance from the camera
                pov_z = rot_z;             
% 
%                 %Print 3D model
%                 printChair(rot_x ,rot_y,pov_z);
%                 axis([-5 5 -5 5  -5 5])      
%                 view(2)
                
                
                x_noise = normrnd(0,0.015,[1,10]);
                y_noise = normrnd(0,0.015,[1,10]);                                
              
                %Project image
                proj_x = ((rot_x))*focal_length+x_noise;
                proj_y = ((rot_y))*focal_length+y_noise;
                
                %Scale it
                proj_x = (proj_x+2)*368/4;
                proj_y = (proj_y+2)*368/4;
                
%                 figure
%                 plot(proj_x,proj_y,'o')
%                 axis([0 368 0 368])
                  
                step = j+1 + (f_count-1)*361 + (i-1)*361*3;
                
                x_in(step,:) = proj_x;
                y_in(step,:) = proj_y;
                z_out(step,:) = rot_z;

%                 figure
%                 x3 = (proj_x - mean(proj_x))./((std(proj_x)+std(proj_y))/2);
%                 y3 = (proj_y - mean(proj_y))./((std(proj_x)+std(proj_y))/2);                   
%                 z3 = pov_z;    
%                                                  
%                 printChair(x3,y3,z3);
%                 axis([-10 10 -10 10 -10 10 ]); 
%                 view(2);

                   
            end
            
            f_count=f_count+1;
    end
            
end


