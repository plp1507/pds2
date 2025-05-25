%% Limpa as variáveis, fecha as figuras correntes e cria o diretório de resultados
clc;
clear all;
close all;
mkdir('Results'); %Diretório para armazenar os resultados

%% Parâmetros de configuração
dataFileName = 'sharky.linear.points'; %sharky.linear.points - sharky.circle.points - sharky.wave.points - sharky.spirals.points
nbrOfNeuronsInEachHiddenLayer = [4 10 5]; %linear:[4] - circle:[10] - wave,spirals:[10 10]
nbrOfOutUnits = 2;
unipolarBipolarSelector = 0; %0 para Unipolar, -1 para Bipolar

learningRate = 0.201;
nbrOfEpochs_max = 1000;

enable_resilient_gradient_descent = 0; %1 para habilitar, 0 para desabilitar
learningRate_plus = 1.2;
learningRate_negative = 0.5;
deltas_start = 0.9;
deltas_min = 10^-6;
deltas_max = 50;

enable_decrease_learningRate = 0; %1 para habilitar o decréscimo da taxa de aprendizagem, 0 para desabilitar
learningRate_decreaseValue = 0.0001;
min_learningRate = 0.05;

enable_learningRate_momentum = 0; %1 para habilitar, 0 para desabilitar
momentum_alpha = 0.05;

draw_each_nbrOfEpochs = 100;

%% Leitura dos dados
importedData = importdata(dataFileName, '\t', 6);
Samples = importedData.data(:, 1:length(importedData.data(1,:))-1);
TargetClasses = importedData.data(:, length(importedData.data(1,:)));
TargetClasses = TargetClasses - min(TargetClasses);
ActualClasses = -1*ones(size(TargetClasses));

%% Calcula o número de nós de ativação de entrada e saída
nbrOfInputNodes = length(Samples(1,:)); %=dimensão para qualquer número de dados de entrada
% nbrOfOutUnits = ceil(log2(length(unique(TargetClasses)))) + !; %Ceil(Log2( Number of Classes ))

nbrOfLayers = 2 + length(nbrOfNeuronsInEachHiddenLayer);
nbrOfNodesPerLayer = [nbrOfInputNodes nbrOfNeuronsInEachHiddenLayer nbrOfOutUnits];

%% Adiciona os Bias como nós com ativação fixa de 1
nbrOfNodesPerLayer(1:end-1) = nbrOfNodesPerLayer(1:end-1) + 1;
Samples = [ones(length(Samples(:,1)),1) Samples];

%% Calcula as saídas (TargetOutputs)
TargetOutputs = zeros(length(TargetClasses), nbrOfOutUnits);
for i=1:length(TargetClasses)
    if (TargetClasses(i) == 1)
        TargetOutputs(i,:) = [1 unipolarBipolarSelector];
    else
        TargetOutputs(i,:) = [unipolarBipolarSelector 1];
    end
end

%% Inicializa as matrizes de pesos aleatórios
Weights = cell(1, nbrOfLayers); %Os pesos que conectam os bias à camada anterior são inúteis, mas isto torna o código mais simples e rápido
Delta_Weights = cell(1, nbrOfLayers);
ResilientDeltas = Delta_Weights; %Passo necessário caso o Gradiente Descendente Resiliente for utilizado
for i = 1:length(Weights)-1
    Weights{i} = 2*rand(nbrOfNodesPerLayer(i), nbrOfNodesPerLayer(i+1))-1;
    Weights{i}(:,1) = 0;
    Delta_Weights{i} = zeros(nbrOfNodesPerLayer(i), nbrOfNodesPerLayer(i+1));
    ResilientDeltas{i} = deltas_start*ones(nbrOfNodesPerLayer(i), nbrOfNodesPerLayer(i+1));
end
Weights{end} = ones(nbrOfNodesPerLayer(end), 1); %Pesos virtuais para os neurônios de saída
Old_Delta_Weights_for_Momentum = Delta_Weights;
Old_Delta_Weights_for_Resilient = Delta_Weights;

NodesActivations = cell(1, nbrOfLayers);
for i = 1:length(NodesActivations)
    NodesActivations{i} = zeros(1, nbrOfNodesPerLayer(i));
end
NodesBackPropagatedErrors = NodesActivations; %Passo necessário para o retrocesso do treinamento por retropropagação do erro

zeroRMSReached = 0;
nbrOfEpochs_done = 0;

%% Treinamento para todos os dados
MSE = -1 * ones(1,nbrOfEpochs_max);
for Epoch = 1:nbrOfEpochs_max

    for Sample = 1:length(Samples(:,1))
        %% Treinamento por retropropagação do erro (backpropagation)
        %Estágio Feedforward
        NodesActivations{1} = Samples(Sample,:);
        for Layer = 2:nbrOfLayers
            NodesActivations{Layer} = NodesActivations{Layer-1}*Weights{Layer-1};
            NodesActivations{Layer} = Activation_func(NodesActivations{Layer}, unipolarBipolarSelector);
            if (Layer ~= nbrOfLayers) %Necessário porque os bias não têm pesos conectados à camada anterior
                NodesActivations{Layer}(1) = 1;
            end
        end

        % Estágio de retropropagação
		% Como os gradientes dos bias são zero, eles não contribuem para as camadas anteriores, nem para a variação dos pesos (delta_weights)
        NodesBackPropagatedErrors{nbrOfLayers} =  TargetOutputs(Sample,:)-NodesActivations{nbrOfLayers};
        for Layer = nbrOfLayers-1:-1:1
            gradient = Activation_func_drev(NodesActivations{Layer+1}, unipolarBipolarSelector);
            for node=1:length(NodesBackPropagatedErrors{Layer}) % Para todos os neurônios na camada atual
                NodesBackPropagatedErrors{Layer}(node) =  sum( NodesBackPropagatedErrors{Layer+1} .* gradient .* Weights{Layer}(node,:) );
            end
        end

        % Cálculo da variação dos pesos (delta_weights) para retropropagação (antes da multiplicação pela taxa de aprendizado)
        for Layer = nbrOfLayers:-1:2
            derivative = Activation_func_drev(NodesActivations{Layer}, unipolarBipolarSelector);
            Delta_Weights{Layer-1} = Delta_Weights{Layer-1} + NodesActivations{Layer-1}' * (NodesBackPropagatedErrors{Layer} .* derivative);
        end
    end

    	%% Aplica o algoritmo de Gradiente Descendente Resiliente e/ou momentum para a variação dos pesos (delta_weights)
    if (enable_resilient_gradient_descent) % Aplica o algoritmo de Gradiente Descendente Resiliente
        if (mod(Epoch,200)==0) %Reseta os Deltas
            for Layer = 1:nbrOfLayers
                ResilientDeltas{Layer} = learningRate*Delta_Weights{Layer};
            end
        end
        for Layer = 1:nbrOfLayers-1
            mult = Old_Delta_Weights_for_Resilient{Layer} .* Delta_Weights{Layer};
            ResilientDeltas{Layer}(mult > 0) = ResilientDeltas{Layer}(mult > 0) * learningRate_plus; % Sem mudança de sinal
            ResilientDeltas{Layer}(mult < 0) = ResilientDeltas{Layer}(mult < 0) * learningRate_negative; % Com mudança de sinal
            ResilientDeltas{Layer} = max(deltas_min, ResilientDeltas{Layer});
            ResilientDeltas{Layer} = min(deltas_max, ResilientDeltas{Layer});

            Old_Delta_Weights_for_Resilient{Layer} = Delta_Weights{Layer};

            Delta_Weights{Layer} = sign(Delta_Weights{Layer}) .* ResilientDeltas{Layer};
        end
    end
    if (enable_learningRate_momentum) %Aplica o momento
        for Layer = 1:nbrOfLayers
            Delta_Weights{Layer} = learningRate*Delta_Weights{Layer} + momentum_alpha*Old_Delta_Weights_for_Momentum{Layer};
        end
        Old_Delta_Weights_for_Momentum = Delta_Weights;
    end
    if (~enable_learningRate_momentum && ~enable_resilient_gradient_descent)
        for Layer = 1:nbrOfLayers
            Delta_Weights{Layer} = learningRate * Delta_Weights{Layer};
        end
    end

    %% Backward Pass Weights Update
    for Layer = 1:nbrOfLayers-1
        Weights{Layer} = Weights{Layer} + Delta_Weights{Layer};
    end

    % Reinicia Delta_Weights para zeros
    for Layer = 1:length(Delta_Weights)
        Delta_Weights{Layer} = 0 * Delta_Weights{Layer};
    end

    %% Decrementa a taxa de aprendizado
    if (enable_decrease_learningRate)
        new_learningRate = learningRate - learningRate_decreaseValue;
        learningRate = max(min_learningRate, new_learningRate);
    end

    %% Avaliação
    for Sample = 1:length(Samples(:,1))
        outputs = EvaluateNetwork(Samples(Sample,:), NodesActivations, Weights, unipolarBipolarSelector);
        bound = (1+unipolarBipolarSelector)/2;
        if (outputs(1) >= bound && outputs(2) < bound)
            ActualClasses(Sample) = 1;
        elseif (outputs(1) < bound && outputs(2) >= bound)
            ActualClasses(Sample) = 0;
        else
            if (outputs(1) >= outputs(2))
                ActualClasses(Sample) = 1;
            else
                ActualClasses(Sample) = 0;
            end
        end
    end

    MSE(Epoch) = sum((ActualClasses-TargetClasses).^2)/(length(Samples(:,1)));
    if (MSE(Epoch) == 0)
        zeroRMSReached = 1;
    end

    %% Visualização
    if (zeroRMSReached || mod(Epoch,draw_each_nbrOfEpochs)==0)
        % Desenha os limites de decisão
        unique_TargetClasses = unique(TargetClasses);
        training_colors = {'y.', 'b.'};
        separation_colors = {'g.', 'r.'};
        subplot(2,1,1);
        cla;
        hold on;
        title(['Limite de decisão no na época ' int2str(Epoch) '. O numero maximo de epocas e ' int2str(nbrOfEpochs_max) '.']);

        margin = 0.05; step = 0.05;
        xlim([min(Samples(:,2))-margin max(Samples(:,2))+margin]);
        ylim([min(Samples(:,3))-margin max(Samples(:,3))+margin]);
        for x = min(Samples(:,2))-margin : step : max(Samples(:,2))+margin
            for y = min(Samples(:,3))-margin : step : max(Samples(:,3))+margin
                outputs = EvaluateNetwork([1 x y], NodesActivations, Weights, unipolarBipolarSelector);
                bound = (1+unipolarBipolarSelector)/2;
                if (outputs(1) >= bound && outputs(2) < bound)
                    plot(x, y, separation_colors{1}, 'markersize', 18);
                elseif (outputs(1) < bound && outputs(2) >= bound)
                    plot(x, y, separation_colors{2}, 'markersize', 18);
                else
                    if (outputs(1) >= outputs(2))
                        plot(x, y, separation_colors{1}, 'markersize', 18);
                    else
                        plot(x, y, separation_colors{2}, 'markersize', 18);
                    end
                end
            end
        end

        for i = 1:length(unique_TargetClasses)
            points = Samples(TargetClasses==unique_TargetClasses(i), 2:end);
            plot(points(:,1), points(:,2), training_colors{i}, 'markersize', 10);
        end
        axis equal;

        % Desenha o gráfico do erro médio quadrático
        subplot(2,1,2);
        MSE(MSE==-1) = [];
        plot([MSE(1:Epoch)]);
        ylim([-0.1 0.6]);
        title('Erro Medio Quadratico');
        xlabel('Epocas');
        ylabel('EMQ');
        grid on;

        saveas(gcf, sprintf('Results//fig%i.png', Epoch),'jpg');
        pause(0.05);
    end
    display([int2str(Epoch) ' Epochs done out of ' int2str(nbrOfEpochs_max) ' Epochs. MSE = ' num2str(MSE(Epoch)) ' Learning Rate = ' ...
        num2str(learningRate) '.']);

    nbrOfEpochs_done = Epoch;
    if (zeroRMSReached)
        saveas(gcf, sprintf('Results//Final Result for %s.png', dataFileName),'jpg');
        break;
    end

end
display(['Mean Square Error = ' num2str(MSE(nbrOfEpochs_done)) '.']);

