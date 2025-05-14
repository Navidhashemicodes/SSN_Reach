function Matrix = mat_generator(values ,  indices, original_dim)

Matrix = zeros(original_dim);

leng = size(indices,1);

t = 0;
for i = 1:leng
    index = indices(i,:);
    t = t+1;
    Matrix(index(1) , index(2)) = values(t);
end


end