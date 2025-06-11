classdef verify_NN


    properties
        True_class
        class_threshold
        model
        model_source
        LB
        de
        indices
        original_dim
        output_dim
        mode
        params
    end

    methods
        function obj = verify_NN(True_class,class_threshold,model,model_source,LB,de,indices,original_dim,output_dim,mode,params)
            obj.True_class = True_class;
            obj.class_threshold = class_threshold;
            obj.model = model;
            obj.model_source = model_source;
            obj.LB = LB;
            obj.de = de;
            obj.indices = indices;
            obj.original_dim = original_dim;
            obj.output_dim = output_dim;
            obj.mode = mode;
            obj.params = params;
        end

        function Matrix = mat_generator_no_third(obj, values )
            
            height = obj.original_dim(2);
            width = obj.original_dim(3);
            n_class = obj.original_dim(1);
            Matrix = zeros(height, width, n_class);

            N_perturbed = size(obj.indices,1);

            t = 0;
            for c = 1: n_class
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

        function needs = Provide(obj)
            
            height = obj.output_dim(2);
            width = obj.output_dim(3);
            n_class = obj.output_dim(1);
            n_channel = obj.original_dim(1);
            N = obj.params.Nt;
            N_dir = obj.params.N_dir;
            trn_batch = obj.params.trn_batch;

            N_perturbed = size(obj.indices , 1);

            rng(0)
            Y = zeros(height,width,n_class,N);
            X = zeros(n_channel*N_perturbed , N);
            tic
            %%%%%%%%%%%%%%
            parfor i=1:N
                disp(i)
                Rand = rand(n_channel*N_perturbed,1);
                Rand_matrix = obj.mat_generator_no_third(Rand);
                d_at = zeros(height,width,n_channel);
                for c=1:n_channel
                    d_at(:,:,c) = obj.de(c) * Rand_matrix(:,:,c) ;
                end
                Inp = single(obj.LB + d_at);
                X(:,i) = Rand;
                Y(:,:,:,i) = obj.forward(Inp);
            end
            %%%%%%%%%%%%%
            train_data_run_1 = toc;
            
            nsf2 = obj.params.num_sample_fittable_in_2GB;
            if N_dir > nsf2
                leng = nsf2;
            else
                leng = N_dir;
            end
            ind = 0;
            if mod(N_dir , leng) == 0
                last_i = N_dir/leng ;
            else
                error('Considering Camvid_dataset, if N_dir is more than 30 it should be devisible by 30.')
            end
            for i=1:last_i
                eval(['Y' num2str(i) ' = Y(:,:,:,ind+1:ind+leng);' ]);
                ind = ind+leng;
            end
            
            cd(obj.params.src_dir)
            Text = ' save("Direction_data.mat" ';
            for i=1:last_i
                Text = [ Text  ' ,"Y' num2str(i) '" ']; %#ok<*AGROW>
            end
            Text = [ Text ');'];
            eval(Text);

            Text = 'clear Y1'; %%% You always need to have something here even if it is nonsense, otherwise it clears everything
            for i=1:last_i
                Text = [ Text  ' Y' num2str(i) '  '];
            end
            eval(Text);
            
            mat_file_path = [obj.params.src_dir '/Direction_data.mat'];

            command = sprintf([ 'python Direction_trainer.py --mat_file_path "%s" '...
                      '--num_files %d --N_dir %d --batch_size %d --height %d --width %d '...
                      '--n_class %d '], ...
                      mat_file_path,  last_i, N_dir , trn_batch, height, width, n_class);

            status = system(command);
            delete('Direction_data.mat')

            n1 = numel(Y(:,:,:,1));
            Y = reshape(Y, [n1 , N]);

            C = 20* (  0.001*(mean(Y,2))   +  (0.05-0.001) * 0.5*( min(Y , [], 2) + max(Y , [] , 2) ));

            % load('directions.mat')

            pyenv
            npz = py.numpy.load('directions.npz');
            % shape = cellfun(@double, cell(npz{'Directions'}.shape));
            Directions_py = py.numpy.array(npz{'Directions'});

            directions_list = cell(Directions_py.tolist());
            rows = cellfun(@(row) double(cellfun(@double, cell(row))), directions_list, 'UniformOutput', false);
            Directions = single(vertcat(rows{:}));

            % Directions = Directions_py.reshape(int64(shape));
            
            % Direction_Training_time = double(npz{'Direction_Training_time'});
            Direction_Training_time = double(npz{'Direction_Training_time'}.tolist());
            clear Directions_py  npz
            delete('directions.npz')


            dYV  = Directions' * (Y - C );
            dims = obj.params.dims;
            epochs = obj.params.epochs;
            save("Reduced_dimension.mat", "dYV", "X", "dims", "epochs");

            system('python Trainer_ReLU.py')

            delete('Reduced_dimension.mat')
            load("trained_relu_weights_2h_norm.mat")
            small_net.weights = {double(W1) , double(W2), double(W3)};
            small_net.biases = {double(b1)' , double(b2)', double(b3)'};
            L = cell(size(W1,1),1);
            L(:) = {'poslin'};
            small_net.layers{1} = L ;
            L = cell(size(W2,1),1);
            L(:) = {'poslin'};
            small_net.layers{2} = L ;
            delete('trained_relu_weights_2h_norm.mat')
            %%%%%%%%%%%%%%%%%%%%%%%%


            dimp = height*width;
            dim2 = height*width*n_class;

            tic

            res_trn = abs( Y - ( Directions * NN_eval(small_net , X)  + C ) );


            threshold_normal = obj.params.threshold_normal;
            res_max = max( res_trn ,[] ,2 );
            indexha = find(res_max < threshold_normal);
            res_max(indexha,1) = threshold_normal;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            trn_time1 = toc;


            clear X  Y  res_trn


            ell = obj.params.rank;
            Ns = obj.params.Ns;
            Nsp = obj.params.Nsp;


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
            test_data_run = zeros(1,Num_chuncks);
            res_test_time = zeros(1,Num_chuncks);
            if remainder ~= 0
                chunck_sizes = [chunck_sizes , remainder]; %#ok<NASGU>
                test_data_run = zeros(1, Num_chuncks+1);
                res_test_time = zeros(1,Num_chuncks+1);
            end



            Rs = zeros(1, Ns);
            ind = 0;

            for nc=1:length(chunck_sizes)

                rng(nc)

                len = chunck_sizes(nc);
                Y_test_nc = zeros(height,width,n_class,len);
                X_test_nc = zeros(n_channel*N_perturbed , len);

                tic
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
                    X_test_nc(:,i) = Rand;
                    Y_test_nc(:,:,:,i) = obj.forward(Inp);
                end
                %%%%%%%%%%%%%

                test_data_run(nc) = toc;



                Y_test = reshape(Y_test_nc, [dim2 , len] );

                clear Y_test_nc

                tic
                res_tst = abs(Y_test - ( Directions * NN_eval( small_net , X_test_nc) + C ));
                Rs(ind+1:ind+len) = max( res_tst ./ res_max);
                res_test_time(nc) = toc;

                clear Y_test X_test_nc res_tst

                ind = ind + len;

            end



            tic
            Rs_sorted = sort(Rs);
            R_star = Rs_sorted(:,ell);
            Conf = R_star*res_max;

            conformal_time = toc;

            figure
            plot(Conf)



            needs = struct;
            needs.Conf = Conf;
            needs.small_net = small_net;
            needs.Directions = Directions;
            needs.C = C;
            needs.train_data_run_1 = train_data_run_1;
            needs.trn_time1 = trn_time1;
            needs.test_data_run = test_data_run;
            needs.res_test_time = res_test_time;
            needs.conformal_time = conformal_time;
            needs.Direction_Training_time = Direction_Training_time;
            needs.Model_training_time = Model_training_time;
            needs.R_star = R_star;
            needs.res_max = res_max;
        end


    

        function [classes, Detection_table] = Mask_titles(obj)

            needs = obj.Provide();

            Conf = needs.Conf;
            N = obj.params.Nt;
            N_dir = obj.params.N_dir;
            Ns = obj.params.Ns;
            R_star = needs.R_star;
            res_max = needs.res_max;
            threshold_normal = obj.params.threshold_normal;
            train_data_run_1 = needs.train_data_run_1;
            trn_time1 = needs.trn_time1;
            test_data_run = needs.test_data_run;
            res_test_time = needs.res_test_time;
            conformal_time = needs.conformal_time;
            Direction_Training_time = needs.Direction_Training_time;
            Model_training_time = needs.Direction_Training_time;
            N_perturbed = size(obj.indices,1);
            height = obj.output_dim(2);
            width = obj.output_dim(3);
            n_class = obj.output_dim(1);
            n_channel = obj.original_dim(1);

            dim2 = height*width*n_class;
            dimp = height*width;
            tic
            H.V = [sparse(dim2,1) speye(dim2)];
            H.C = sparse(1,dim2);
            H.d = 0;
            H.predicate_lb = -Conf;
            H.predicate_ub =  Conf;
            H.dim = dim2;
            H.nVar= dim2;

            I = Star();
            I.V = [0.5*ones(n_channel*N_perturbed,1)  eye(n_channel*N_perturbed)];
            I.C = zeros(1,n_channel*N_perturbed);
            I.d = 0;
            I.predicate_lb = -0.5*ones(n_channel*N_perturbed,1);
            I.predicate_ub =  0.5*ones(n_channel*N_perturbed,1);
            I.dim =  n_channel*N_perturbed;
            I.nVar = n_channel*N_perturbed;



            Principal_reach = ReLUNN_Reachability_starinit(I, needs.small_net, 'approx-star');
            Surrogate_reach = affineMap(Principal_reach , needs.Directions , needs.C);

            %%%%%%%%%%%%%%
            reachability_time = toc;

            clear Principal_reach


            tic
            %%%%%%%%%%%%%%
            P_Center = sparse(double(Surrogate_reach.V(:,1)));
            P_lb = double([Surrogate_reach.predicate_lb ; H.predicate_lb]);
            P_ub = double([Surrogate_reach.predicate_ub ; H.predicate_ub]);


            P_V = [double(Surrogate_reach.V(:,2:end))   double(H.V(:,2:end))];
            Lb = P_Center + 0.5*(P_V + abs(P_V))*P_lb + 0.5*(P_V - abs(P_V))*P_ub;
            Ub = P_Center + 0.5*(P_V + abs(P_V))*P_ub + 0.5*(P_V - abs(P_V))*P_lb;


            Lb_pixels = reshape(Lb , [dimp, n_class]);
            Ub_pixels = reshape(Ub , [dimp, n_class]);
            %%%%%%%%%%%%%%
            projection_time = toc;


            clear  Surrogate_reach

            classes = cell(height,width);

            %%%  0 means: background   and    1 means: lung

            for i=1:height
                for j = 1:width
                    t = (j-1)*height+i;
                    [L_star, my_class] = max(Lb_pixels(t,:));
                    class_members = [];
                    for k = 1:n_class
                        if L_star <= Ub_pixels(t,k)
                            class_members = [class_members , k];
                        end
                    end
                    classes{i,j} = class_members;
                end
            end

            True_class = obj.True_class;

            Detection_table = cell(height, width);
            robust = 0 ;
            nonrobust = 0;
            unknown = 0;
            attacked = N_perturbed;
            for i =1:height
                for j=1:width
                    if (length(classes{i,j}) == 1) %#ok<ISCL>
                        if classes{i,j} == True_class{i,j}
                            robust = robust + 1;
                            Detection_table{i,j} = 'robust';
                        else
                            nonrobust = nonrobust + 1;
                            Detection_table{i,j} = 'nonrobust';
                        end
                    else
                        if ismember(True_class{i,j} , classes{i,j} )
                            unknown = unknown + 1;
                            Detection_table{i,j} = 'unknown';
                        else
                            nonrobust = nonrobust + 1;
                            Detection_table{i,j} = 'nonrobust';
                        end
                    end
                end
            end

            dim_pic = height*width;

            RV = 100* robust / dim_pic;
            disp(['Number of Robust pixels: ' num2str(robust)])
            disp(['Number of non-Robust pixels: ' num2str(nonrobust)])
            disp(['Number of unknown pixels: ' num2str(unknown)])
            disp(['RV value: ' num2str(RV)])

            guarantee = obj.params.guarantee;
            ell = obj.params.rank;
            Failure_chance_of_guarantee =  betacdf( guarantee , ell , Ns+1 -ell);

            disp([' Pr[    Pr[    RV_value is ' num2str(RV) '%  ]  >  '  num2str(guarantee) '  ]  >  '  num2str( 1- Failure_chance_of_guarantee)   ])

            verification_runtime = needs.train_data_run_1 + needs.trn_time1 + sum(needs.test_data_run) + sum(needs.res_test_time) + ...
                needs.conformal_time + reachability_time + projection_time + needs.Direction_Training_time + needs.Model_training_time;

            disp([' The verification Runtime is: ' num2str(verification_runtime/60) ' minutes.'   ])

            name = ['CI_result_ReLU_relaxed_eps_' num2str(obj.params.delta_rgb) '_Nperturbed_' num2str(N_perturbed) 'image_' obj.params.image_name ...
                    'model_' obj.params.model_name '.mat'];
            de = obj.de; %#ok<PROP>
            cd(obj.params.current_dir)
            save(name, "robust", "nonrobust", "attacked", "unknown", "True_class", "classes", "Detection_table", "Conf", ...
                "N", "N_dir", "de", "ell", "Lb_pixels", "Ub_pixels", "Ns", "R_star", ...
                "res_max", "RV", "threshold_normal" , "verification_runtime", ...
                "train_data_run_1", "trn_time1", "test_data_run", "res_test_time", ...
                "conformal_time" , "reachability_time", "projection_time", "Direction_Training_time", "Model_training_time");
        end
    end
end