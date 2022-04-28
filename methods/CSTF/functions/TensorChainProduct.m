function X = TensorChainProduct(X,U,list)
for i = 1 : numel(list)
    X = TensorProduct(X,U{list(i)},list(i));
end