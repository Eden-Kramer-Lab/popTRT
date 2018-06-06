%% GoF for two mark-ovarlapping cells with self inhibition and one direction cross-excitation 
clear all; rng(33);
% Set simulation parameters

n=1e4;
sds = .3; sdm = .3; % sd for location state and mark state
 
sigi1 = inv([.5 0;0 sds^2]); m1 = [-2 11]; sigi2 = inv([.5 0; 0 sds^2]); m2 = [2 12];
dm = .01; ms = 9:dm:14; dt=1e-3; dx = .1; xs = -5:dx:5;
dm1 = .1*1e4/n; ms1 = 9:dm1:14;
alpha = .98; sd=.3; a1 = log(150); a2 = a1;
sxker = .1; smker = .2; 
 

% Simulate training data
xtrain(1)=1; spiketrain(1)=0; Imark(1)=0; Imark1(1)=0;
for i=2:n,
    xtrain(i) = alpha*xtrain(i-1)+sd*normrnd(0,1);  % AR(1) process, X_{t+1} = alpha*X_t + e_t
    c1(i) = exp(a1-(xtrain(i)-m1(1))^2*sigi1(1,1)/2); c2(i) = exp(a2-(xtrain(i)-m2(1))^2*sigi2(1,1)/2);
    H1 = find(spiketrain(1:i-1).*(1-Imark(1:i-1))); H2 = find(spiketrain(1:i-1).*(Imark(1:i-1)));
    inh1(i) = prod(1-exp(-(i-H1).^2/2/196)); % 14^2 = 196
    ex1(i) = sum(0+300*exp(-(i-10-H2).^2/2/4)); inh2(i) = prod(1-exp(-(i-H2).^2/2/196));
    Lambda = (c1(i)*inh1(i)+ex1(i)*inh1(i)+c2(i)*inh2(i));
    spiketrain(i) = poissrnd(Lambda*dt)>0;
    mark1 = normrnd(m1(2),sdm); mark2 = normrnd(m2(2),sdm); which = binornd(1,(c1(i)*inh1(i)+ex1(i)*inh1(i))/Lambda); marktrain(i) = which*mark1+(1-which)*mark2;
    Imark1(i) = (marktrain(i)>11.5);
    Imark(i) = 1- which;
end;
ind1 = intersect(find(spiketrain),find(1-Imark)); % indicator for spikes from neuron 1 
ind2 = intersect(find(spiketrain),find(Imark)); % indicator for spikes from neuron 2 
ind = find(spiketrain>0); size(ind)
figure; subplot(211); plot(1:n,xtrain,'k',ind1,xtrain(ind1),'r.',ind2,xtrain(ind2),'b.'); xlabel('Time'); ylabel('x values');
subplot(212); plot(ind1,marktrain(ind1),'r.',ind2,marktrain(ind2),'b.'); xlabel('Time'); ylabel('marks');
%figure(2);subplot(211); hist(marktrain(ind1));xlim([10 13]);subplot(212); hist(marktrain(ind2));xlim([10 13]);



% Time rescaling based on true model
LambdaIntMarks = cumsum(((c1.*inh1+ex1.*inh1)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m1(2),sdm)+((c2.*inh2)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m2(2),sdm)); % LambdaIntMarks is of size (time_steps, spike counts)
RescaledSpikeTimes = LambdaIntMarks([ind1 ind2]+([0:length(ind)-1]*length(xtrain))); % size: (1, n_spikes)
% Calculate the right boudary of rescaled region.
LambdaInt = sum( ((c1.*inh1+ex1.*inh1)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m1(2),sdm)+((c2.*inh2)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m2(2),sdm));
RescaledT = linspace(0,max(LambdaInt),10*n/1e4*length(xtrain));   % we times 10 here to make sure at most 1 spike per bin
dtau = RescaledT(2)-RescaledT(1);   % rescaled time bin size here;
RescaledSpikeTrain = hist(RescaledSpikeTimes,RescaledT);
for t=1:length(RescaledT), 
    Occupancy(t) = sum(LambdaInt>RescaledT(t))*dm*dt*dtau;  % Calculate the Occupancy here
end;

figure; subplot(231); plot(RescaledSpikeTimes,marktrain([ind1 ind2]),'.',LambdaInt,ms); ylabel('marks'); xlabel('Rescaled Time');
Os = hist(marktrain(ind),ms1); % Observed spikes in subregions; ms1 are the bin centers
Es = LambdaInt*dt*dm;
Es1 = zeros(size(ms1));
for i=1:size(ms1, 2)
   idx = ms <= ms1(i)+dm1/2 & ms > ms1(i)-dm1/2;
   Es1(i) = sum(Es(idx));
end;
Eind = find(Es1>5); % expected spikes in subregions
p=1-chi2cdf(sum((Os(Eind)-Es1(Eind)).^2./Es1(Eind)),length(Eind)-1); title(strcat('True model, p = ',num2str(p)));
subplot(234); %KSPlot(Occupancy,RescaledSpikeTrain);
n1 = sum(RescaledSpikeTrain);  % number of spikes
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);

test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf); % KS test for rescaled ISIs 
%[h, p] = kstest2(temp, exprnd(1,n1,1)); 
plot(expcdf(x1,1), f, 'b'); line([0 1], [0 1],'Color','black');  line([0 1],[0 1]+1.36/sqrt(n1),'Color','black','LineStyle','--');line([0 1], [0 1]-1.36/sqrt(n1),'Color','black','LineStyle','--'); axis([0 1 0 1]);
ylabel('Model CDF'); xlabel('Empirical CDF'); title(strcat('True model KS, p = ',num2str(p)));



% Time rescaling based on marked model missing history
LambdaIntMarks = cumsum(((c1)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m1(2),sdm)+((c2)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m2(2),sdm));
RescaledSpikeTimes = LambdaIntMarks([ind1 ind2]+([0:length(ind)-1]*length(xtrain)));
 
LambdaInt = sum( ((c1)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m1(2),sdm)+((c2)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m2(2),sdm));
RescaledT = linspace(0,max(LambdaInt),10*n/1e4*length(xtrain));  %% add 10*
dtau = RescaledT(2)-RescaledT(1);
RescaledSpikeTrain = hist(RescaledSpikeTimes,RescaledT);
for t=1:length(RescaledT), Occupancy(t) = sum(LambdaInt>RescaledT(t))*dm*dt*dtau; end;
 
subplot(232); plot(RescaledSpikeTimes,marktrain([ind1 ind2]),'.',LambdaInt,ms); xlabel('Rescaled Time'); ylabel('marks');
Os = hist(marktrain(ind),ms1); % Observed spikes in subregions; ms1 are the bin centers
Es = LambdaInt*dt*dm;
Es1 = zeros(size(ms1));
for i=1:size(ms1, 2)
   idx = ms <= ms1(i)+dm1/2 & ms > ms1(i)-dm1/2;
   Es1(i) = sum(Es(idx));
end;
Eind = find(Es1>5); % expected spikes in subregions
p=1-chi2cdf(sum((Os(Eind)-Es1(Eind)).^2./Es1(Eind)),length(Eind)-1); title(strcat('Missing history, p = ',num2str(p)));

subplot(235); %KSPlot(Occupancy,RescaledSpikeTrain);
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);
test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf);p % KS test for rescaled ISIs 
n1 = sum(RescaledSpikeTrain);
%[h, p] = kstest2(temp, exprnd(1,n1,1)); 
plot(expcdf(x1,1), f, 'b'); line([0 1], [0 1],'Color','black');  line([0 1],[0 1]+1.36/sqrt(n1),'Color','black','LineStyle','--');line([0 1], [0 1]-1.36/sqrt(n1),'Color','black','LineStyle','--'); axis([0 1 0 1]);
ylabel('Model CDF'); xlabel('Empirical CDF');title(strcat('Missing history KS, p = ',num2str(p)));
 
 
% Based on Sorted model
ind2 = find(spiketrain>0&Imark1==1); ind1 = find(spiketrain>0&Imark1==0);
for i=2:n,
    H1 = find(spiketrain(1:i-1).*(1-Imark1(1:i-1))); H2 = find(spiketrain(1:i-1).*(Imark1(1:i-1)));
    inh1(i) = prod(1-exp(-(i-H1).^2/2/196)); ex1(i) = sum(0+300*exp(-(i-10-H2).^2/2/4)); inh2(i) = prod(1-exp(-(i-H2).^2/2/196));
end;
lam2 = cumsum((c2.*inh2)'); lam1 = cumsum((c1.*inh1+ex1.*inh1)');  
res1 = lam1(ind1); res2 = lam2(ind2);  % rescaled spike time for neuron 1 and 2
RescaledT = linspace(0,max([lam1(end) lam2(end)]),10*n/1e4*length(xtrain)); % linear segments in rescaled time space,
%%%%%%%%%%%%% something to improve here, add 10*
dtau = RescaledT(2)-RescaledT(1); % bin size
RescaledSpikeTrain = hist([res1; res2], RescaledT); 
Occupancy = ((lam1(end)>RescaledT)+(lam2(end)>RescaledT))*dt*dtau; % only take 2 different values
 
subplot(233); plot([res1'; res1'],[1*ones(size(res1')); 2*ones(size(res1'))],'b',[res2'; res2'],[0*ones(size(res2')); 1*ones(size(res2'))],'r'); title('Sorted spikes');
subplot(236); %KSPlot(Occupancy,RescaledSpikeTrain); xlabel('Rescaled Time');
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);
test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf); % KS test for rescaled ISIs 
n1 = sum(RescaledSpikeTrain);
%[h, p] = kstest2(temp, exprnd(1,n1,1)); 
plot(expcdf(x1,1), f, 'b'); line([0 1], [0 1],'Color','black');  line([0 1],[0 1]+1.36/sqrt(n1),'Color','black','LineStyle','--');line([0 1], [0 1]-1.36/sqrt(n1),'Color','black','LineStyle','--'); axis([0 1 0 1]);
ylabel('Model CDF'); xlabel('Empirical CDF');title(strcat('True model KS, p = ',num2str(p)));
 



%figure(3);plot(1:10^4,Imark.*spiketrain,1:10^4,Imark1.*spiketrain, 'Color','r');
%figure; plot(cumsum(RescaledSpikeTrain-Occupancy))


%% GoF for two mark-ovarlapping cells with self inhibition and one direction cross-excitation 
clear all; rng(33);
% Set simulation parameters

n=1e4;
sds = .3; sdm = .3; % sd for location state and mark state
 
sigi1 = inv([.5 0;0 sds^2]); m1 = [-2 11]; sigi2 = inv([.5 0; 0 sds^2]); m2 = [2 12];
dm = .01; ms = 9:dm:14; dt=1e-3; dx = .1; xs = -5:dx:5;
dm1 = .1*1e4/n; ms1 = 9:dm1:14;
alpha = .98; sd=.3; a1 = log(150); a2 = a1;
sxker = .1; smker = .2; 
 

% Simulate training data
xtrain(1)=1; spiketrain(1)=0; Imark(1)=0; Imark1(1)=0;
for i=2:n,
    xtrain(i) = alpha*xtrain(i-1)+sd*normrnd(0,1);  % AR(1) process, X_{t+1} = alpha*X_t + e_t
    c1(i) = exp(a1-(xtrain(i)-m1(1))^2*sigi1(1,1)/2); c2(i) = exp(a2-(xtrain(i)-m2(1))^2*sigi2(1,1)/2);
    H1 = find(spiketrain(1:i-1).*(1-Imark(1:i-1))); H2 = find(spiketrain(1:i-1).*(Imark(1:i-1)));
    inh1(i) = prod(1-exp(-(i-H1).^2/2/196)); % 14^2 = 196
    ex1(i) = sum(0+300*exp(-(i-10-H2).^2/2/4)); inh2(i) = prod(1-exp(-(i-H2).^2/2/196));
    Lambda = (c1(i)*inh1(i)+ex1(i)*inh1(i)+c2(i)*inh2(i));
    spiketrain(i) = poissrnd(Lambda*dt)>0;
    mark1 = normrnd(m1(2),sdm); mark2 = normrnd(m2(2),sdm); which = binornd(1,(c1(i)*inh1(i)+ex1(i)*inh1(i))/Lambda); marktrain(i) = which*mark1+(1-which)*mark2;
    Imark1(i) = (marktrain(i)>11.5);
    Imark(i) = 1- which;
end;
ind1 = intersect(find(spiketrain),find(1-Imark)); % indicator for spikes from neuron 1 
ind2 = intersect(find(spiketrain),find(Imark)); % indicator for spikes from neuron 2 
ind = find(spiketrain>0); size(ind)
%figure; subplot(211); plot(1:n,xtrain,'k',ind1,xtrain(ind1),'r.',ind2,xtrain(ind2),'b.'); xlabel('Time'); ylabel('x values');
%subplot(212); plot(ind1,marktrain(ind1),'r.',ind2,marktrain(ind2),'b.'); xlabel('Time'); ylabel('marks');
%figure(2);subplot(211); hist(marktrain(ind1));xlim([10 13]);subplot(212); hist(marktrain(ind2));xlim([10 13]);


%
% % Estimate the parameters
% x = xtrain(spiketrain > 0); size(x);
% mark = marktrain(spiketrain >0); size(mark);
% X = [x;mark]'; size(X);
% options = statset('Display','final');
% GMModel = fitgmdist(X,2,'CovarianceType','diagonal','Options',options);
% GMModel.mu
% % Another method is from VLFeat package
% import vlfeat;
% [means, covariances, priors] = vl_gmm(X, 2)  % vl_gmm from VLFeat package
% 


% Set up based on estimated parameters
sdm1 = 0.25; sdm2 = 0.52;
sigi1 = inv([0.62^2 0;0 sds^2]); m1 = [-1.44 11.01]; sigi2 = inv([0.79^2 0; 0 sds^2]); m2 = [1.65 11.60];
sxker = .1; smker = .2;

for i=2:n,
    c1(i) = exp(a1-(xtrain(i)-m1(1))^2*sigi1(1,1)/2); c2(i) = exp(a2-(xtrain(i)-m2(1))^2*sigi2(1,1)/2);
    H1 = find(spiketrain(1:i-1).*(1-Imark(1:i-1))); H2 = find(spiketrain(1:i-1).*(Imark(1:i-1)));
    inh1(i) = prod(1-exp(-(i-H1).^2/2/196)); % 14^2 = 196
    ex1(i) = sum(0+300*exp(-(i-10-H2).^2/2/4)); inh2(i) = prod(1-exp(-(i-H2).^2/2/196));
end;

% Time rescaling based on true model
LambdaIntMarks = cumsum(((c1.*inh1+ex1.*inh1)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m1(2),sdm1)+((c2.*inh2)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m2(2),sdm2)); % LambdaIntMarks is of size (time_steps, spike counts)
RescaledSpikeTimes = LambdaIntMarks([ind1 ind2]+([0:length(ind)-1]*length(xtrain))); % size: (1, n_spikes)
% Calculate the right boudary of rescaled region.
LambdaInt = sum( ((c1.*inh1+ex1.*inh1)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m1(2),sdm1)+((c2.*inh2)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m2(2),sdm2));
RescaledT = linspace(0,max(LambdaInt),10*n/1e4*length(xtrain));   % we times 10 here to make sure at most 1 spike per bin
dtau = RescaledT(2)-RescaledT(1);   % rescaled time bin size here;
RescaledSpikeTrain = hist(RescaledSpikeTimes,RescaledT);
for t=1:length(RescaledT), 
    Occupancy(t) = sum(LambdaInt>RescaledT(t))*dm*dt*dtau;  % Calculate the Occupancy here
end;

figure; subplot(231); plot(RescaledSpikeTimes,marktrain([ind1 ind2]),'.',LambdaInt,ms); ylabel('marks'); xlabel('Rescaled Time');
Os = hist(marktrain(ind),ms1); % Observed spikes in subregions; ms1 are the bin centers
Es = LambdaInt*dt*dm;
Es1 = zeros(size(ms1));
for i=1:size(ms1, 2)
   idx = ms <= ms1(i)+dm1/2 & ms > ms1(i)-dm1/2;
   Es1(i) = sum(Es(idx));
end;
Eind = find(Es1>5); % expected spikes in subregions
p=1-chi2cdf(sum((Os(Eind)-Es1(Eind)).^2./Es1(Eind)),length(Eind)-1); title(strcat('True model, p = ',num2str(p)));
subplot(234); %KSPlot(Occupancy,RescaledSpikeTrain);
n1 = sum(RescaledSpikeTrain);  % number of spikes
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);

test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf); % KS test for rescaled ISIs 
%[h, p] = kstest2(temp, exprnd(1,n1,1)); 
plot(expcdf(x1,1), f, 'b'); line([0 1], [0 1],'Color','black');  line([0 1],[0 1]+1.36/sqrt(n1),'Color','black','LineStyle','--');line([0 1], [0 1]-1.36/sqrt(n1),'Color','black','LineStyle','--'); axis([0 1 0 1]);
ylabel('Model CDF'); xlabel('Empirical CDF'); title(strcat('True model KS, p = ',num2str(p)));


% Time rescaling based on marked model missing history, estimated
LambdaIntMarks = cumsum(((c1)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m1(2),sdm1)+((c2)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m2(2),sdm2));
RescaledSpikeTimes = LambdaIntMarks([ind1 ind2]+([0:length(ind)-1]*length(xtrain)));
 
LambdaInt = sum( ((c1)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m1(2),sdm1)+((c2)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m2(2),sdm2));
RescaledT = linspace(0,max(LambdaInt),10*n/1e4*length(xtrain));  %% add 10*
dtau = RescaledT(2)-RescaledT(1);
RescaledSpikeTrain = hist(RescaledSpikeTimes,RescaledT);
for t=1:length(RescaledT), Occupancy(t) = sum(LambdaInt>RescaledT(t))*dm*dt*dtau; end;
 
subplot(232); plot(RescaledSpikeTimes,marktrain([ind1 ind2]),'.',LambdaInt,ms); xlabel('Rescaled Time'); ylabel('marks');
Os = hist(marktrain(ind),ms1); % Observed spikes in subregions; ms1 are the bin centers
Es = LambdaInt*dt*dm;
Es1 = zeros(size(ms1));
for i=1:size(ms1, 2)
   idx = ms <= ms1(i)+dm1/2 & ms > ms1(i)-dm1/2;
   Es1(i) = sum(Es(idx));
end;
Eind = find(Es1>5); % expected spikes in subregions
p=1-chi2cdf(sum((Os(Eind)-Es1(Eind)).^2./Es1(Eind)),length(Eind)-1); title(strcat('Missing history, p = ',num2str(p)));

subplot(235); %KSPlot(Occupancy,RescaledSpikeTrain);
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);
test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf);p % KS test for rescaled ISIs 
n1 = sum(RescaledSpikeTrain);
%[h, p] = kstest2(temp, exprnd(1,n1,1)); 
plot(expcdf(x1,1), f, 'b'); line([0 1], [0 1],'Color','black');  line([0 1],[0 1]+1.36/sqrt(n1),'Color','black','LineStyle','--');line([0 1], [0 1]-1.36/sqrt(n1),'Color','black','LineStyle','--'); axis([0 1 0 1]);
ylabel('Model CDF'); xlabel('Empirical CDF');title(strcat('Missing history KS, p = ',num2str(p)));
 


% Based on Sorted model, estimated
ind2 = find(spiketrain>0&Imark1==1); ind1 = find(spiketrain>0&Imark1==0);
for i=2:n,
    H1 = find(spiketrain(1:i-1).*(1-Imark1(1:i-1))); H2 = find(spiketrain(1:i-1).*(Imark1(1:i-1)));
    inh1(i) = prod(1-exp(-(i-H1).^2/2/196)); ex1(i) = sum(0+300*exp(-(i-10-H2).^2/2/4)); inh2(i) = prod(1-exp(-(i-H2).^2/2/196));
end;
lam2 = cumsum((c2.*inh2)'); lam1 = cumsum((c1.*inh1+ex1.*inh1)');  
res1 = lam1(ind1); res2 = lam2(ind2);  % rescaled spike time for neuron 1 and 2
RescaledT = linspace(0,max([lam1(end) lam2(end)]),10*n/1e4*length(xtrain)); % linear segments in rescaled time space,
%%%%%%%%%%%%% something to improve here, add 10*
dtau = RescaledT(2)-RescaledT(1); % bin size
RescaledSpikeTrain = hist([res1; res2], RescaledT); 
Occupancy = ((lam1(end)>RescaledT)+(lam2(end)>RescaledT))*dt*dtau; % only take 2 different values
 
subplot(233); plot([res1'; res1'],[1*ones(size(res1')); 2*ones(size(res1'))],'b',[res2'; res2'],[0*ones(size(res2')); 1*ones(size(res2'))],'r'); title('Sorted spikes');
subplot(236); %KSPlot(Occupancy,RescaledSpikeTrain); xlabel('Rescaled Time');
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);
test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf); % KS test for rescaled ISIs 
n1 = sum(RescaledSpikeTrain);
%[h, p] = kstest2(temp, exprnd(1,n1,1)); 
plot(expcdf(x1,1), f, 'b'); line([0 1], [0 1],'Color','black');  line([0 1],[0 1]+1.36/sqrt(n1),'Color','black','LineStyle','--');line([0 1], [0 1]-1.36/sqrt(n1),'Color','black','LineStyle','--'); axis([0 1 0 1]);
ylabel('Model CDF'); xlabel('Empirical CDF');title(strcat('True model KS, p = ',num2str(p)));
 


%% GoF for two mark-ovarlapping cells with self inhibition and one direction cross-excitation 
clear all; 
counts = 200;
N1 = [500 750 1000 2000 5000 7500 10000 15000 20000];  % Define number time steps
KS_p = zeros(size(N1,2), counts, 3);  
Pearson_p = zeros(size(N1,2), counts, 2);


for l=1:size(N1,2)
    for k=1:size(KS_p, 2)
rng(k);
% Set simulation parameters

n=N1(l);
sds = .3; sdm = .3; % sd for location state and mark state
 
sigi1 = inv([.5 0;0 sds^2]); m1 = [-2 11]; sigi2 = inv([.5 0; 0 sds^2]); m2 = [2 12];
dm = .01; ms = 9:dm:14; dt=1e-3; dx = .1; xs = -5:dx:5;
dm1 = 0.1*1e4/n; ms1 = 9:dm1:14;
alpha = .98; sd=.3; a1 = log(150); a2 = a1;
sxker = .1; smker = .2; 
 
% Simulate training data
xtrain(1)=1; spiketrain(1)=0; Imark(1)=0; Imark1(1)=0;
for i=2:n,
    xtrain(i) = alpha*xtrain(i-1)+sd*normrnd(0,1);
    c1(i) = exp(a1-(xtrain(i)-m1(1))^2*sigi1(1,1)/2); c2(i) = exp(a2-(xtrain(i)-m2(1))^2*sigi2(1,1)/2);
    H1 = find(spiketrain(1:i-1).*(1-Imark(1:i-1))); H2 = find(spiketrain(1:i-1).*(Imark(1:i-1)));
    inh1(i) = prod(1-exp(-(i-H1).^2/2/196)); % 14^2 = 196
    ex1(i) = sum(0+300*exp(-(i-10-H2).^2/2/4)); inh2(i) = prod(1-exp(-(i-H2).^2/2/196));
    Lambda = (c1(i)*inh1(i)+ex1(i)*inh1(i)+c2(i)*inh2(i));
    spiketrain(i) = poissrnd(Lambda*dt)>0;
    mark1 = normrnd(m1(2),sdm); mark2 = normrnd(m2(2),sdm); which = binornd(1,(c1(i)*inh1(i)+ex1(i)*inh1(i))/Lambda); marktrain(i) = which*mark1+(1-which)*mark2;
    Imark1(i) = (marktrain(i)>11.5);
    Imark(i) = 1- which;
end;
ind1 = intersect(find(spiketrain),find(1-Imark)); % indicator for spikes from neuron 1 
ind2 = intersect(find(spiketrain),find(Imark)); % indicator for spikes from neuron 2 
ind = find(spiketrain>0); size(ind);

% Time rescaling based on true model
LambdaIntMarks = cumsum(((c1.*inh1+ex1.*inh1)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m1(2),sdm)+((c2.*inh2)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m2(2),sdm));
RescaledSpikeTimes = LambdaIntMarks([ind1 ind2]+([0:length(ind)-1]*length(xtrain)));
 
LambdaInt = sum( ((c1.*inh1+ex1.*inh1)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m1(2),sdm)+((c2.*inh2)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m2(2),sdm));
RescaledT = linspace(0,max(LambdaInt),10*n/1e4*length(xtrain));   %%%%%%%%%%%%% something to improve here, add 10*
dtau = RescaledT(2)-RescaledT(1);   % rescaled time bin size here;
RescaledSpikeTrain = hist(RescaledSpikeTimes,RescaledT);
for t=1:length(RescaledT), 
    Occupancy(t) = sum(LambdaInt>RescaledT(t))*dm*dt*dtau; 
end;

Os = hist(marktrain(ind),ms1); % Observed spikes in subregions.
Es = LambdaInt*dt*dm;
Es1 = zeros(size(ms1));
for i=1:size(ms1, 2)
   idx = ms <= ms1(i)+dm1/2 & ms > ms1(i)-dm1/2;
   Es1(i) = sum(Es(idx));
end;
Eind = find(Es1>5); % expected spikes in subregions
p=1-chi2cdf(sum((Os(Eind)-Es1(Eind)).^2./Es1(Eind)),length(Eind)-1); 
Pearson_p(l,k,1) = p;
n1 = sum(RescaledSpikeTrain);  % number of spikes
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);
test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf); % KS test for rescaled ISIs 
KS_p(l,k,1) = p;


% Time rescaling based on marked model missing history
LambdaIntMarks = cumsum(((c1)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m1(2),sdm)+((c2)'*ones(size(ind))).*normpdf(ones(size(xtrain'))*marktrain([ind1 ind2]),m2(2),sdm));
RescaledSpikeTimes = LambdaIntMarks([ind1 ind2]+([0:length(ind)-1]*length(xtrain)));
 
LambdaInt = sum( ((c1)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m1(2),sdm)+((c2)'*ones(size(ms))).*normpdf(ones(size(xtrain'))*ms,m2(2),sdm));
RescaledT = linspace(0,max(LambdaInt),10*n/1e4*length(xtrain));  %% add 10*
dtau = RescaledT(2)-RescaledT(1);
RescaledSpikeTrain = hist(RescaledSpikeTimes,RescaledT);
for t=1:length(RescaledT), Occupancy(t) = sum(LambdaInt>RescaledT(t))*dm*dt*dtau; end;
 
Os = hist(marktrain(ind),ms1); % Observed spikes in subregions.
Es = LambdaInt*dt*dm;
Es1 = zeros(size(ms1));
for i=1:size(ms1, 2)
   idx = ms <= ms1(i)+dm1/2 & ms > ms1(i)-dm1/2;
   Es1(i) = sum(Es(idx));
end;
Eind = find(Es1>5); % expected spikes in subregions
p=1-chi2cdf(sum((Os(Eind)-Es1(Eind)).^2./Es1(Eind)),length(Eind)-1); 
Pearson_p(l,k,2) = p;
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);
test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf); % KS test for rescaled ISIs 
KS_p(l,k,2) = p; 

 
% Based on Sorted model
ind2 = find(spiketrain>0&Imark1==1); ind1 = find(spiketrain>0&Imark1==0);
for i=2:n,
    H1 = find(spiketrain(1:i-1).*(1-Imark1(1:i-1))); H2 = find(spiketrain(1:i-1).*(Imark1(1:i-1)));
    inh1(i) = prod(1-exp(-(i-H1).^2/2/196)); ex1(i) = sum(0+300*exp(-(i-10-H2).^2/2/4)); inh2(i) = prod(1-exp(-(i-H2).^2/2/196));
end;
lam2 = cumsum((c2.*inh2)'); lam1 = cumsum((c1.*inh1+ex1.*inh1)');  
res1 = lam1(ind1); res2 = lam2(ind2);  % rescaled spike time for neuron 1 and 2
RescaledT = linspace(0,max([lam1(end) lam2(end)]),10*n/1e4*length(xtrain)); % linear segments in rescaled time space,
%%%%%%%%%%%%% something to improve here, add 10*
dtau = RescaledT(2)-RescaledT(1); % bin size
RescaledSpikeTrain = hist([res1; res2], RescaledT); 
Occupancy = ((lam1(end)>RescaledT)+(lam2(end)>RescaledT))*dt*dtau; % only take 2 different values
 
temp = cumsum(Occupancy); temp = temp(RescaledSpikeTrain>0); temp = [temp(1) diff(temp)]; [f,x1] = ecdf(temp);
test_cdf = makedist('Exponential','mu',1); [h, p] = kstest(temp,'CDF',test_cdf); % KS test for rescaled ISIs 
KS_p(l,k,3) = p;
%[h, p] = kstest2(temp, exprnd(1,n1,1)); 
    end;
end;

figure(1);
subplot(121); plot(N1, mean(Pearson_p(:,:,1)<0.05,2), 'r',N1, mean(Pearson_p(:,:,2)<0.05,2), 'b');
ylabel('Fraction rejected'); xlabel('Expected spike counts');title('Uniform test');
subplot(122); plot(N1, mean(KS_p(:,:,1)<0.05,2), 'r',N1, mean(KS_p(:,:,2)<0.05,2), 'b',N1, mean(KS_p(:,:,3)<0.05,2), 'y');
ylabel('Fraction rejected'); xlabel('Expected spike counts');title('KS test');









