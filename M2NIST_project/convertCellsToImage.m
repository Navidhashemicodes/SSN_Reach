function img = convertCellsToImage(cell1, cell2)
    % Define the size of the image
    [rows, cols] = size(cell1);
    img = uint8(zeros(rows, cols, 3)); % Initialize an RGB image
    
    % Define color map
    color_map = containers.Map({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, ...
        {[0, 0, 139], [0, 100, 0], [173, 216, 230], [139, 69, 19], [144, 238, 144], ...
         [152, 251, 152], [255, 255, 0], [128, 128, 128], [255, 140, 0], [255, 165, 0], [0, 0, 0]});

%     [0, 0, 139] → Dark Blue
% 
%     [0, 100, 0] → Dark Green
% 
%     [173, 216, 230] → Light Blue
% 
%     [139, 69, 19] → Saddle Brown
% 
%     [144, 238, 144] → Light Green
% 
%     [152, 251, 152] → Pale Green
% 
%     [255, 255, 0] → Yellow
% 
%     [128, 128, 128] → Gray
% 
%     [255, 140, 0] → Dark Orange
% 
%     [255, 165, 0] → Orange
% 
%     [0, 0, 0] → Black
    
    % Iterate over each pixel location
    for i = 1:rows
        for j = 1:cols

            if length(cell2{i, j}) ~= 1 || isempty(cell1{i, j})
                error('Double-check the True-class')
            end

            if isequal(cell1{i, j}, cell2{i, j})
                
                img(i, j, :) =  color_map(cell2{i,j});
                    
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
