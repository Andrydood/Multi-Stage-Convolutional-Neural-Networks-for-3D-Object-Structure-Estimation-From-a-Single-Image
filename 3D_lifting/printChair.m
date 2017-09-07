function printChair( x,y,z )
    

        hold off
        
        for i = 1:10
            
            plot3(x(i),y(i),z(i),'.','MarkerSize', 30);
            hold on
        end
        i = 1;
        j = 5;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');

        i = 2;
        j = 6;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');
        
        i = 4;
        j = 8;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black'); 
        
        i = 3;
        j = 7;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');  
        
        i = 5;
        j = 6;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');

        i = 6;
        j = 7;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');
        
        i = 7;
        j = 8;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black'); 
        
        i = 8;
        j = 5;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');
        
        i = 8;
        j = 9;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');
        
        i = 9;
        j = 10;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black'); 
        
        i = 10;
        j = 7;
        plot3([x(i),x(j)],[y(i),y(j)],[z(i),z(j)],'-','MarkerSize', 30,'color','black');  
        
end

