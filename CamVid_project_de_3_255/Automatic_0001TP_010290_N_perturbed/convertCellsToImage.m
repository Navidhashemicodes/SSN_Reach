function img = convertCellsToImage(cell1, cell2)
    % Define the size of the image
    [rows, cols] = size(cell1);
    img = uint8(zeros(rows, cols, 3)); % Initialize an RGB image
    
    % Define color map
    color_map = containers.Map({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ...
        {[0	128	192], [128	0	0], [64	0	128], [192	192	128], [64	64	128], ...
         [64	64	0], [128	64	128], [0	0	192], [192	128	128], [128	128	128], [128	128	0], [0 , 0, 0]});



% class 1  Bicyclist	 0	    128	192	1
% class 2  Building	     128	0	0	1
% class 3  Car	         64	    0	128	1
% class 4  Column_Pole   192	192	128	1
% class 5  Fence	     64	    64	128	1
% class 6  Pedestrian	 64	    64	0	1
% class 7  Road	         128	64	128	1
% class 8  Sidewalk	     0	    0	192	1
% class 9  SignSymbol	 192	128	128	1
% class 10 Sky	         128	128	128	1
% class 11 Tree	         128	128	0	1
% class 12 Void	         0  	0	0	1



    % Iterate over each pixel location
    for i = 1:rows
        for j = 1:cols


            if length(cell2{i, j}) ~= 1 || isempty(cell1{i, j})
                error('Double-check the True-class')
            end
            
            if isequal(cell1{i,j}, cell2{i,j})

                img(i, j, :) = color_map(cell2{i,j});
            
            else
                if length(cell1{i,j}) > 1

                    img(i, j, :) = [255, 0, 0]; % Red if the class is unknown
                
                else
                
                    img(i, j, :) = [139, 0, 0]; % Dark-Red if the classes are different but known
                
                end
            end
        end
    end
    
    % Display the image
    figure;
    imshow(img, 'InitialMagnification', 'fit');
end
