classdef Conservatism_analysis


    properties

        model
        model_source
        LB
        de
        LB_out
        UB_out
        indices
        original_dim
        output_dim
        params
    end

    methods
        function obj = Conservatism_analysis(model,model_source,LB,de,LB_out,UB_out,indices,original_dim,output_dim,params)
            obj.model = model;
            obj.model_source = model_source;
            obj.LB = LB;
            obj.de = de;
            obj.LB_out = LB_out;
            obj.UB_out = UB_out;
            obj.indices = indices;
            obj.original_dim = original_dim;
            obj.output_dim = output_dim;
            obj.params = params;
        end

        function Matrix = mat_generator_no_third(obj, values )
            
            height = obj.original_dim(2);
            width = obj.original_dim(3);
            n_channel = obj.original_dim(1);
            Matrix = zeros(height, width, n_channel);

            N_perturbed = size(obj.indices,1);

            t = 0;
            for c = 1: n_channel
                for i = 1:N_perturbed
                    index = obj.indices(i,:);
                    t = t+1;
                    Matrix(index(1) , index(2) , c) = values(t);
                end
            end


        end

        function out = forward(obj, x)

            switch obj.model_source
                case 'onnx'
                    [out, ~] = obj.model(x, obj.params.model_params, 'InputDataPermutation', 'auto');

                case 'SeriesNetwork'
                    out = obj.model.predict(x);

                otherwise
                    error("Unknown model source: " + obj.model_source + ". We only cover onnx or SeriesNetwork.");
            end
        end

        function [empirical_miscoverage, Y_min, Y_max] = conservatism(obj)
            
            Ns = obj.params.Ns;
            Nsp = obj.params.Nsp;
            Seed0 = obj.params.Seed0;
            height = obj.output_dim(2);
            width = obj.output_dim(3);
            n_class = obj.output_dim(1);
            n_channel = obj.original_dim(1);
            N_perturbed = size(obj.indices , 1);


            if Ns > Nsp
                thelen = Nsp;
            else
                thelen = Ns;
            end


            if Ns > thelen
                chunck_size = thelen;
                Num_chuncks = floor(Ns / chunck_size);
                remainder = mod(Ns , chunck_size);
            else
                chunck_size = Ns;
                Num_chuncks = 1;
                remainder = 0;
            end
            chunck_sizes = chunck_size * ones(1, Num_chuncks);
            if remainder ~= 0
                chunck_sizes = [chunck_sizes , remainder]; %#ok<NASGU>
            end

            outsides = zeros(1, Ns);
            ind = 0;
            
            dim2 = height*width*n_class;
            
            for nc=1:length(chunck_sizes)

                rng(Seed0+nc)

                len = chunck_sizes(nc);
                Y_test_nc = zeros(height,width,n_class,len);

                
                %%%%%%%%%%%%%%
                parfor i=1:len
                    disp(['part ' num2str(nc) ' number ' num2str(i)])
                    Rand = rand(n_channel*N_perturbed,1);
                    Rand_matrix = obj.mat_generator_no_third(Rand);
                    d_at = zeros(height,width,n_channel);
                    for c=1:n_channel
                        d_at(:,:,c) = obj.de(c) * Rand_matrix(:,:,c) ;
                    end
                    Inp = obj.LB + d_at;
                    Y_test_nc(:,:,:,i) = obj.forward(Inp);
                end
                %%%%%%%%%%%%%

                Y_test = reshape(Y_test_nc, [dim2 , len] );

                clear Y_test_nc

                outsides(ind+1 : ind+len) = any((Y_test < obj.LB_out) | (Y_test > obj.UB_out), 1);

                if nc == 1
                    Y_max = max(Y_test, [], 2);
                    Y_min = min(Y_test, [], 2);
                else
                    Y_max = max(Y_max, max(Y_test, [], 2));
                    Y_min = min(Y_min, min(Y_test, [], 2));
                end

                ind = ind + len;

            end

        empirical_miscoverage = mean(outsides);
        
        end
    end
end
