% Carga de datos
wine = load('wine.data.txt');
% Ordenamiento aleatorio de filas
[m,n] = size(wine);
wine = wine(randperm(m),:);
% Feature scalling y normalización
X = wine(:,2:n);
y = wine(:,1);
n = n-1;
avg_X = zeros(n,1);
std_X = zeros(n,1);
for i = 1:n
   avg_X(i) = mean(X(:,i));
   std_X(i) = std(X(:,i));
end
X_norm = zeros(size(X));
for i = 1:n
   X_norm(:,i) = (X(:,i)-avg_X(i))./std_X(i);
end
% Entrenamiento y validación con cross validation
k = 10;
cp = cvpartition(y, 'Kfold',k);
order = unique(y);

f = @(X_train,y_train,X_test,y_test)...
    confusionmat(y_test,svmpredict(y_test, X_test,svmtrain(y_train, X_train, '-t 3')));

cfMat = crossval(f,X_norm,y,'partition',cp);
cfMat = reshape(sum(cfMat),3,3)
performance = sum(diag(cfMat))/m*100;
fprintf("Se clasificó correctamente al %f%% de observaciones.\n",performance);
