filename = 'spec.txt';  % Replace with your actual file path
fid = fopen(filename, 'r');

% Initialize containers
lb = zeros(3072, 1);
ub = zeros(3072, 1);

while ~feof(fid)
    line = fgetl(fid);
    tokens = regexp(line, '\(assert \((<=|>=) X_(\d+) ([\d\.\-eE]+)\)\)', 'tokens');
    if ~isempty(tokens)
        tokens = tokens{1};
        direction = tokens{1};         % '<=' or '>='
        index = str2double(tokens{2}); % variable index
        value = str2double(tokens{3}); % bound value

        if strcmp(direction, '<=')     % Upper bound
            ub(index + 1) = value;
        elseif strcmp(direction, '>=')
            lb(index + 1) = value;
        end
    end
end

fclose(fid);

save('Bounds.mat' , 'lb' , "ub")