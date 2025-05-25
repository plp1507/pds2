%% Função de Ativação dos neurônios
function fx = Activation_func(x, unipolarBipolarSelector)
    if (unipolarBipolarSelector == 0)
        fx = 1./(1 + exp(-x)); %Unipolar
    else
        fx = -1 + 2./(1 + exp(-x)); %Bipolar
    end
end
