clear
clc

for i=2:20
    for j=1:6
        delete('directions.mat')
        delete('Direction_data.mat')
        delete('Reduced_dimension.mat')
        delete('trained_relu_weights_2h_norm.mat')

        model_name = 'm2nist_dilated_72iou_24layer.mat';
        Text = sprintf("verify_M2NIST_on_CNNS( [32-10 , 42-10], %d, %d, '%s', 1, 3, 2000, 2000, 100000 , 99999, 0.9999  )",j*17,i, model_name);
        eval(Text);
    end
end

for i=1:20
    for j=1:6
        delete('directions.mat')
        delete('Direction_data.mat')
        delete('Reduced_dimension.mat')
        delete('trained_relu_weights_2h_norm.mat')

        model_name = 'm2nist_75iou_transposedcnn_avgpool.mat';
        Text = sprintf("verify_M2NIST_on_CNNS( [32-10 , 42-10], %d, %d, '%s', 2, 3, 2000, 2000, 100000 , 99999, 0.9999  )",j*17,i, model_name);
        eval(Text)
    end
end


for i=1:20
    for j=1:6
        delete('directions.mat')
        delete('Direction_data.mat')
        delete('Reduced_dimension.mat')
        delete('trained_relu_weights_2h_norm.mat')

        model_name = 'm2nist_62iou_dilatedcnn_avgpool.mat';
        Text = sprintf("verify_M2NIST_on_CNNS( [32-10 , 42-10], %d, %d, '%s', 3, 3, 2000, 2000, 100000 , 99999, 0.9999  )",j*17,i, model_name);
        eval(Text)
    end
end