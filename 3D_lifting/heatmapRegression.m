clear
close all

SIZE = 200;

in = readNPY('../../data/input.npy');
in = reshape(in,[SIZE,SIZE,3]);
gt  = readNPY('../../data/ground_truth.npy');
gt = reshape(gt,[SIZE,SIZE,11]);
out  = readNPY('../../data/output.npy');
out = reshape(out,[SIZE,SIZE,11]);

imshow(in/255);

out_inv = abs(out(:,:,11)-1);

seg = out_inv>0.7;

keypoints = zeros(200,200,6);

queue = zeros(40000,2);

for kp = 1:6
    
    keypointh = 0;
    keypointw = 0;
    
    for h = 1:200
        
        for w=1:200
            
            if seg(h,w) == 1
                
                keypointh = h;
                keypointw = w;
                
                break
                
            end
            if keypointh~= 0 
                break
            end
        end
        if keypointh~= 0 
           break
        end
    end
    
    keypoints(keypointh,keypointw,kp) = 1;
    seg(keypointh,keypointw) = 0;
    done = 0;
    qindex = 1;
    queue(qindex,:) = [keypointh,keypointw];
    currentq = 1;
    
    for test=1:40000
    
        qh = queue(currentq,1);
        qw = queue(currentq,2);
        
        if~(qh>199 | qh<2)
            if~(qw>199 | qw<2)
                                
                if(seg(qh+1,qw+1)==1)
                    keypoints(qh+1,qw+1,kp) = 1;
                    seg(qh+1,qw+1) = 0;
                    queue(qindex,:) = [qh+1,qw+1];
                    qindex = qindex+1;
                end
                
                if(seg(qh-1,qw+1)==1)
                    keypoints(qh-1,qw+1,kp) = 1;
                    seg(qh-1,qw+1) = 0;
                    queue(qindex,:) = [qh-1,qw+1];
                    qindex = qindex+1;
                end
                
                if(seg(qh+1,qw-1)==1)
                    keypoints(qh+1,qw-1,kp) = 1;
                    seg(qh+1,qw-1) = 0;
                    queue(qindex,:) = [qh+1,qw-1];
                    qindex = qindex+1;
                end
                
                if(seg(qh-1,qw-1)==1)
                    keypoints(qh-1,qw-1,kp) = 1;
                    seg(qh-1,qw-1) = 0;
                    queue(qindex,:) = [qh-1,qw-1];;
                    qindex = qindex+1;
                end
                
                if(seg(qh,qw+1)==1)
                    keypoints(qh,qw+1,kp) = 1;
                    seg(qh,qw+1) = 0;
                    queue(qindex,:) = [qh,qw+1];
                    qindex = qindex+1;
                end
                
                if(seg(qh-1,qw)==1)
                    keypoints(qh-1,qw,kp) = 1;
                    seg(qh-1,qw) = 0;
                    queue(qindex,:) = [qh-1,qw];
                    qindex = qindex+1;
                end
                
                if(seg(qh+1,qw)==1)
                    keypoints(qh+1,qw,kp) = 1;
                    seg(qh+1,qw) = 0;
                    queue(qindex,:) = [qh+1,qw];
                    qindex = qindex+1;
                end
                
                if(seg(qh,qw-1)==1)
                    keypoints(qh,qw-1,kp) = 1;
                    seg(qh,qw-1) = 0;
                    queue(qindex,:) = [qh,qw-1];
                    qindex = qindex+1;
                end
                
            end 
        end
        currentq = currentq+1;
    end
    
end

for i =1:6
    
    figure
    imshow(keypoints(:,:,i))
    
end

outKeypoints = zeros(6,2);

for i=1:6
    xmean = 0;
    ymean = 0;
    count = 1;
    
    for x = 1:200
        
        for y = 1:200
            
            if keypoints(x,y,i) == 1
                
                xmean = xmean + x;
                ymean = ymean + y;
                
                count = count +1;
                
            end
            
        end
        
    end
    xmean = round(xmean/count);
    ymean = round(ymean/count);
    
    outKeypoints(i,:) = [xmean,ymean];
    
end

output_keypoints = zeros(200,200,10);

topleft = 1;
topright = 2;
midleft = 4;
midright = 3;
botleft = 6;
botright = 5;

size = 5;

output_keypoints(outKeypoints(botleft,1)+size,outKeypoints(botleft,2)-size,1) = 1;
output_keypoints(outKeypoints(botleft,1)-size,outKeypoints(botleft,2)+size,4) = 1;
output_keypoints(outKeypoints(botright,1)-size,outKeypoints(botright,2)-size,3) = 1;
output_keypoints(outKeypoints(botright,1)+size,outKeypoints(botright,2)+size,2) = 1;

output_keypoints(outKeypoints(midleft,1)+size,outKeypoints(midleft,2)-size,5) = 1;
output_keypoints(outKeypoints(midleft,1)-size,outKeypoints(midleft,2)+size,8) = 1;
output_keypoints(outKeypoints(midright,1)-size,outKeypoints(midright,2)-size,7) = 1;
output_keypoints(outKeypoints(midright,1)+size,outKeypoints(midright,2)+size,6) = 1;

output_keypoints(outKeypoints(topleft,1),outKeypoints(topleft,2),9) = 1;
output_keypoints(outKeypoints(topright,1),outKeypoints(topright,2),10) = 1;



figure
imshow(sum(output_keypoints(:,:,:),3))

% for i = 1:4
%     figure
%     imshow(output_keypoints(:,:,i))
% end

writeNPY(output_keypoints,'heatmaps.npy');



