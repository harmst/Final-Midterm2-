clear; close all; clc
%Training Labels
TrainingLabels = fopen('train-labels.idx1-ubyte');
A = fread(TrainingLabels,inf,'uint8');
A_prime = A(9:end,:);
ST = fclose(TrainingLabels);

%Training Images
TrainingImages = fopen('train-images.idx3-ubyte');
A2 = fread(TrainingImages,inf,'uint8');
A2_prime = A2(17:end,:);
ST2 = fclose(TrainingImages);

%Testing Labels
TestingLabels = fopen('t10k-labels.idx1-ubyte');
A3 = fread(TrainingLabels,inf,'uint8');
A3_prime = A3(9:end,:);
ST3 = fclose(TrainingLabels);

%Training Images
TestingImages = fopen('t10k-images.idx3-ubyte');
A4 = fread(TrainingImages,inf,'uint8');
A4_prime = A4(17:end,:);
ST4 = fclose(TrainingImages);

% Testing reading in images and double checking against labels
ImageIndexNum = 40874; 
offset = (ImageIndexNum-1)*(28*28);

% Test Method 2
pic_test = A2_prime(offset+1:offset+28*28,1);
pic_test = reshape(pic_test,28,28);
pic_test = pic_test.';

% print image and corresponding label from label vector
% pcolor was not working, it inverts/mirrors the image
figure(1)
imagesc(pic_test);
colormap gray
A_prime((offset/(28*28))+1)
title('Test Image')

%% Resize image matrices
A2_prime = reshape(A2_prime, [28*28, 60000]);
A4_prime = reshape(A4_prime, [28*28, 10000]);

%% Denote y_j and combine into B (label matrix)

%Training Labels
C = eye(10);
L1 = length(A_prime);
B1 = zeros(10,L1);    %initializing B1
for i = 1:L1
    if A_prime(i) == 0
        j1 = 10;
    else 
        j1 = A_prime(i);
    end
    B1(:,i) = C(:,j1);
end

%Testing Labels
C = eye(10);
L2 = length(A3_prime);
B2 = zeros(10,L2);    %initializing B2
for i = 1:L2
    if A3_prime(i) == 0
        j2 = 10;
    else 
        j2 = A3_prime(i);
    end
    B2(:,i) = C(:,j2);
end

%% Solvers A*X=B

%Training Data
X1 = A2_prime;
B1 = B1;
X2 = A4_prime;
B2 = B2;

%% PsuedoInverse
a1 = B1*pinv(X1);
save('PsuedoInverse.mat', 'a1')


%% Backslash
a2 = B1/X1;
save('Backslash.mat', 'a2')

%% Lasso 1
lambda = 0.01;
for i = 1:10
    a3 = lasso(X1.', B1(i,:), 'lambda', 0.01);
    a3 = imresize(a3,[784,1]);
    a3_tr(:,i) = a3;
end
a3 = a3_tr';
save('Lasso1.mat','a3')

%% Lasso 2
lambda = 0.001;
for i = 1:10
    a4 = lasso(X1.', B1(i,:),'lambda', 0.001);
    a4 = imresize(a4,[784,1]);
    a4_tr(:,i) = a4;
end
a4 = a4_tr';
save('Lasso2.mat','a4')

%% Putting Back Together - Checking Accuracy
ea1 = matfile('PsuedoInverse.mat');
a1 = ea1.a1;
ea2 = matfile('Backslash.mat');
a2 = ea2.a2;
ea3 = matfile('Lasso1.mat');
a3 = ea3.a3;
ea4 = matfile('Lasso2.mat');
a4 = ea4.a4;

b1 = a1*X1;
b2 = a2*X1;
b3 = a3*X1;
b4 = a4*X1;

%Accuracy
er1 = norm(b1-B1,2)/norm(B1,2);
acc1 = sqrt(er1);
er2 = norm(b2-B1, 2)/norm(B1,2);
acc2 = sqrt(er2);
er3 = norm(b3-B1,2)/norm(B1,2);
acc3 = sqrt(er3);
er4 = norm(b4-B1,2)/norm(B1,2);
acc4 = sqrt(er4);


%% Using Sparity determine and rank important pixels - Lasso

for j = 1:784
    norm1 = norm(a3(:,j),1);     %l1 norm - sparsity
    norm2(:,j) = norm1;
end
a3_im = reshape(norm2.', [28,28]);

figure(3)
imagesc(a3_im)
colorbar
title('Important Pixels')


%% Apply Important Pixels to Test Data
%Using the a from the initial Lasso a3 - lambda = 0.01

B2tilde = a3*X2;     %new label matrix to compare to B2
[M,l] = max(B2tilde); 

for k = 1:10000
    if l(k) == 10;
        l(k) = 0;
    else
        l(k) = l(k);
    end
end

bool = l' == A3_prime;
ave = sum(bool);
dist = length(l);
B2tildeacc = ave/dist;



%% Redo with each digit
%Using part 2 (important pixels) 
%find the most important pixels for each digit

%preallocate
ave2 = [];
bool2 = [];
p2 = [];

for n = 1:10
    for j = 1:784
        norm1 = norm(a3(n,j),2);   
        norm2(n,j) = norm1;
    end
    
    figure;
    imagesc(reshape(norm2(n,:).',28,28));
    
  
    %Using part 3 - apply important pixels to test data
    B2tilde2 = a3*X2;
    [M,l2] = max(B2tilde2); 

    for k = 1:10000
         if l2(k) == 10;
            l2(k) = 0;
         else
            l2(k) = l2(k);
         end
    end

    bool2(:,n) = l2(n)' == A3_prime;
    ave2(:,n) = sum(bool2(:,n));   %how many are predicted
    p2(:,n) = ave2(:,n)/length(l2(:,n));

    if n == 10
        q = 0;
    else
        q = n;
    end
    count = A3_prime == q;
    count(n) = sum(count);
    p2err = norm(B2tilde2-B2,2)/norm(B2,2);
    p2acc = sqrt(p2err);
    d = abs(p2(n)-count(n));
    d(n) = d;
    finalacc = 100*p2(n)/count(n);
    finalacc(n) = finalacc; 
    
    
end

figure;
bar(ave2(1:10))
title('Test Images Accurately Predicted')
ylim([0 10000])













    

