function [UTotStats, UseStats, p, FinSprSD, UseGameN] = ExcelTest()

%clear
%clc
%close all
format long
fileID = fopen('TeamNamesBB.txt','r');
TeamNames = textscan(fileID,'%s','Delimiter','\n'); % extract stats from team's txtfile
fclose(fileID);

year = '2017';

count = 0;
numTeams = 351;
UseSOS = zeros(1,numTeams);
UseSpread = zeros(1,numTeams);
UseGameN = zeros(1,numTeams);
UseScore = zeros(1,numTeams);

TotStats = zeros(10500,21);

for h=2:1:2 %% loop that goes through all teams to collect stats and opp stats
    
    Name = [TeamNames{1}{h},'BBXS2018.mat'];
    
    [ExcTest] = load(Name);
        
        %determine teams season info
        A(:,1) = ExcTest.ExcNum(:,1); % team score
        A(:,2) = ExcTest.ExcNum(:,2); % opponent score
        A(:,4) = ExcTest.ExcNum(:,6); % team three pointers made
        
        if (size(char(ExcTest.ExcText(:,1)),2)) > 1
            A(:,3) = ' ';
            D = char(ExcTest.ExcText(:,1));% Opponent's opponents
        else
            A(:,3) = char(ExcTest.ExcText(:,1));
            D = char(ExcTest.ExcText(:,2));% Opponent's opponents
        end
        
        OFGA = ExcTest.ExcNum(:,21); % opponent field goal attempts; used to match game numbers with opponent
        S=A(:,1); % extract team score from matrix A
        Snew=S(2:length(S)); % team scores minus the first game
        OS=A(:,2); % extract opp score from matrix A
        Spread = S - OS; % calculate spread of game
        SpreadN = Spread(2:length(Spread)); % spread minus the first game
        Loc=A(:,3); % extract location from matrix A
        Threes = A(:,4); % extract team three pointers made
        OThrees = ExcTest.ExcNum(:,23);
        %max = numel(D)/length(D);
        GameNum = 2:1:length(S); % team game num from 2-last game
        UsedGames = GameNum - 1;
        avSc = zeros(1,length(Snew)); % allocate space for team average score
        avOSc = zeros(1,length(Snew)); % allocate space for opp average score
        avSpread = zeros(1,length(Snew));
        TotAvSc = zeros(1,length(Snew)); 
        PctPaint = zeros(1,length(Snew));
        PctOPaint = zeros(1,length(Snew));
        Pct3 = zeros(1,length(Snew)); % allocate space for percent of points from 3s
        
        for i =1:1:length(Snew) % determines average team and opp score based on history before that game
            TotPoints = sum(S(1:i));
            TotOPoints = sum(OS(1:i));
            SThrees = sum(Threes(1:i));
            SOThrees = sum(OThrees(1:i));
            PctPaint(i) = (TotPoints - 3*SThrees)/TotPoints;
            PctOPaint(i) = (TotOPoints - 3*SOThrees)/TotOPoints;
            avSc(i) = mean(S(1:i));
            avSpread(i) = mean(Spread(1:i));
            avOSc(i) = mean(OS(1:i));
            TotAvSc(i) = mean(S(1:length(Snew)));
            Pct3(i) = (mean(Threes(1:i))*3)/avSc(i);
            % determine location
            if Loc(i+1) == ' '
                Loc(i) = 1;
            elseif Loc(i+1) == 'N'
                   Loc(i) = 0;
            else
                Loc(i) = -1;
            end
        end

        %%extra stats for use
        UseSpread(h) = mean(Spread);
        UseGameN(h) = length(GameNum) + 1;
        UseScore(h) = Snew(end);
        
        %%useful team stats      
        newStats = [Snew avSc' avOSc' Loc(1:(end-1)) UsedGames' Pct3' TotAvSc' SpreadN avSpread' PctPaint' PctOPaint'];
        
        %gather opponent info from season
        newOppStats = zeros(length(S)-1,10);
        for j=2:1:length(S) 
        Dnew(1,:) = deblank(D(j,1:1:end)); % get Game Opp name from getting rid of trailing blank space

   Name = [Dnew,'BBXS2018.mat'];
    
    [ExcTest2] = load(Name);
    
        A2(:,1) = ExcTest2.ExcNum(:,1);% opp score
        A2(:,2) = ExcTest2.ExcNum(:,2); % opp's opp score
        A2(:,5) = ExcTest2.ExcNum(:,23); % opp three pointers allowed
        
        if (size(char(ExcTest2.ExcText(:,1)),2)) > 1
            A2(:,3) = ' ';
            D2 = char(ExcTest2.ExcText(:,1));% Opponent's opponents
        else
            A2(:,3) = char(ExcTest2.ExcText(:,1));
            D2 = char(ExcTest2.ExcText(:,2));% Opponent's opponents
         end
        
        FGA = ExcTest2.ExcNum(:,4); % opp field goals attempted used to match game number
        S2=A2(:,1); % extact opp score from matrix A2
        OS2=A2(:,2); % extract opp's opp score from matrix A2
        %Loc2=A2(:,3); % extract opp locatoin from matrix A2
        %OGN = A2(:,4); % extract opp game numbers from matrix A2
        Threes2 = A2(:,5); % extract opp three pointers allowed
        ThreesO = ExcTest2.ExcNum(:,6); % extract team three pointers made
        OThreesO = A2(:,5);
        
        for k =1:1:length(S2) % match games with team to determine how many games opp played before playing team
            if S2(k) == OS(j) && OS2(k) == S(j) && FGA(k) == OFGA(j) && k == 1
                AvOS = 0; % avg opp score before game
                AvOOS = 0; % avg opp's opp score before game
                avSpread2 = 0;
                OTotPoints = 0;
                TotOOS = mean(S2(1:length(S2-1)));
                OPct3 = 0;
                OGU = k-1;
                break;
            end
            if S2(k) == OS(j) && OS2(k) == S(j) && FGA(k) == OFGA(j)
                AvOS =  mean(S2(1:k-1)); % avg opp score before game
                AvOOS = mean(OS2(1:k-1)); % avg opp's opp score before game
                OTotPoints = sum(S2(1:k-1));
                OTotOPoints = sum(OS2(1:k-1));
                avSpread2 = AvOS - AvOOS;
                TotOOS = mean(S2(1:length(S2-1)));
                OPct3 = (mean(Threes2(1:k-1))*3)/AvOOS;
                OGU = k-1;
                SThreesO = sum(ThreesO(1:k-1));
                SOThreesO = sum(OThreesO(1:k-1));
                OPctPaint = (OTotPoints - 3*SThreesO)/OTotPoints;
                OPctOPaint = (OTotOPoints - 3*SOThreesO)/OTotOPoints;
                break;
            end
        end
        PointDif = zeros(1,length(S)-1);
        
       % Previous Team Opponents 
            for m=1:1:(j-1)           
                Fnew(1,:) = deblank(D(m,1:1:end));
                
                Name = [Fnew,'BBXS2018.mat'];
    
                [ExcTest3] = load(Name);
                
                A3(:,1) = ExcTest3.ExcNum(:,1); % opp score
                A3(:,2) = ExcTest3.ExcNum(:,2); % opp's opp score
                S3=A3(:,1); % extract opp score from matrix A3
                OS3=A3(:,2); % extract opp's opp score from matrix A3
                
                if length(S3) >= j
                    PointDif(m) = mean(S3(1:(j-1))-OS3(1:(j-1)));
                else
                    PointDif(m) = mean(S3(1:end)-OS3(1:end));
                end
                
               clear -regexp ^Fnew ^ExcTest3 ^A3 ^OS3 
            end
            
           UseSOS(h) = mean([PointDif avSpread2]); 
            
            OPointDif = zeros(1,length(OGU));
            if OGU == 0
                OPointDif = 0;
            else
            
            for n=1:1:OGU
            Gnew(1,:) = deblank(D2(n,1:1:end)); % get Game Opp name from getting rid of trailing blank space

                Name = [Gnew,'BBXS2018.mat'];
    
                [ExcTest4] = load(Name);
                
                A4(:,1) = ExcTest4.ExcNum(:,1); % opp score
                A4(:,2) = ExcTest4.ExcNum(:,2); % opp's opp score
                S4=A4(:,1); % extact opp score from matrix A3
                OS4=A4(:,2); % extract opp's opp score from matrix A3
                
                if length(S4) >= OGU
                    OPointDif(n) = mean(S4(1:OGU)-OS4(1:OGU));
                else
                    OPointDif(n) = mean(S4(1:end)-OS4(1:end));
                end
                
                clear -regexp ^Gnew ^ExcTest4 ^A4 ^OS4
            end
            end
            
            OSOS = mean(OPointDif);
            SOS = mean(PointDif(1:j-1));
            
        %%useful opposing teams stats
        newOppStats(j-1,:) = [AvOS AvOOS TotOOS SOS OSOS OGU OPct3 avSpread2 OPctPaint OPctOPaint];     

        clear -regexp ^Dnew ^ExcTest2 ^A2 ^Snew ^S2 ^OS2 ^GameNum ^PointDif ^OPointDif ^AvOS ^AvOOS
        
        end
         
        clear -regexp ^Loc ^A ^ExcTest ^PctPaint ^PctOPaint

        % combine stats once all data is entered
        
            TotStats(count+1:count+length(avSc),:) = [newStats newOppStats];
            count = count + length(avSc);

        clear -regexp ^Loc ^A ^ExcTest ^newStats ^newOppStats ^S ^OS
       
end

% UTotStats = TotStats(1,:);
% %get rid of '0' scores - put UTotStats into separtate file
% for i =2:1:length(TotStats)
%     if TotStats(i,12) ~= 0
%         UTotStats = [UTotStats;TotStats(i,:)];
%     end
% end

    ULoad = load('UTotStats.mat');
    
    UTotStats = ULoad.UTotStats;

%% Statistical Analysis

% close all
% clc
% clear
x=1:1:150;
Games = 1:1:39;    
numStats = 21;
numTeams = 351;

% using avgscore, location, both sos, and opp opp's avgscore for games only before game played

fileID = fopen('TotStats2016.txt','r');
TT = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter','\t'); % extract stats from team's txtfile
fclose(fileID);

for i =1:1:numStats %% create individual cell vectors
    H(:,i)=TT{i};
end

fileID = fopen('TotStats2017.txt','r');
TT2 = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter','\t'); % extract stats from team's txtfile
fclose(fileID);

for i =1:1:numStats %% create individual cell vectors
    H2(:,i)=TT2{i};
end

fileID = fopen('2018Temp.txt','r');
TT3 = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter','\t'); % extract stats from team's txtfile
fclose(fileID);

for i =1:1:numStats %% create individual cell vectors
    H3(:,i)=TT3{i};
end

H = [UTotStats; H; H2]; % eventually put UTotStats in

GaNu = H(:,5); % game numbers
Perform = zeros(length(H),1);
AvPerform = zeros(length(H),1);
TTotDiff = zeros(length(H),1);
TTotPrediction = zeros(length(H), 1);
numVars1 = 5;

%{
% allocate space for standard deviation and variable coefficients
SD = zeros(1,Games(end));
V1 = zeros(1,Games(end));
V2 = zeros(1,Games(end));
V3 = zeros(1,Games(end));
V4 = zeros(1,Games(end));
V5 = zeros(1,Games(end));
V6 = zeros(1,Games(end));
%V7 = zeros(1,Games(end));

% loop that creates array with the variables and results, cycles for each
% game number
for j = 1:1:39 %H(end,5)
    k=1;
    %count = 0;
for i=1:1:length(H)
    if H(i,5) == j
       P(k,:) = [H(i,2) H(i,4) H(i,11) H(i,13) H(i,14)];% H(i,6)-H(i,14)];
       R(k,:) = H(i,1);
       Nums(k) = i;
       k = k+1;
    end
end

if j < 37
    Predictors = [ones(length(P),1) P]; % final matrix
elseif j == 37
    Predictors = [ones(size(P,1),1) P];
elseif j > 37
    Predictors = [ones(size(P,1),1) P];
end

Stats = regress(R,Predictors); % stats on matrix

% set variable coefficients
V1(j) = Stats(1);
V2(j) = Stats(2);
V3(j) = Stats(3);
V4(j) = Stats(4);
V5(j) = Stats(5);
V6(j) = Stats(6);

lm = fitlm(P,R,'linear'); % multivariable regression

Prediction = Stats(1) + Stats(2).*P(:,1) + Stats(3).*P(:,2) + Stats(4).*P(:,3) + Stats(5).*P(:,4) + Stats(6).*P(:,5);% + Stats(7).*P(:,6); % predicted score 
Diff = Prediction - R; % difference between prediction and actual results

TTotDiff(Nums') = Diff(:,1);

%{
% produce total difference matrix
if j == 1
    TotDiff = Diff;
else
    TotDiff = [TotDiff; Diff];
end
%}

SD(j) = sqrt(sum((Diff.^2))/(length(Diff)-1)); % standard deviation based on exact coefficients for each game number

%figure(j)
%histogram(RoundDif)
%{
figure(j)
plot(Prediction,R,'g.')
hold on
plot(x,x,'r')
%}
%{
figure(j+50)
histogram(Diff)
%}

TTotPrediction(Nums') = Prediction(:,1);

%{
if j == 1
    TotPrediction = Prediction;
else
    TotPrediction = [TotPrediction; Prediction];
end
%}

clear -regexp ^P ^R ^Prediction ^Stats ^Diff ^Nums ^RVP ^DifAv ^UsableNums ^AdjUsableNums ^AdjRVP ^AdjDifAv ^AdjPred 

end

figure(1)
plot(TTotPrediction,H(:,1),'g.')

figure(3)
grid on
plot(SD,'g')

GamesU = (1:30);
% plots for each variable coeffients with functions for best fit
figure(37)
plot(Games, V1,'r.')
hold on
p(1,:) = polyfit(GamesU,V1(1:30),2);
plot(GamesU, p(1,3) + p(1,2)*GamesU + p(1,1)*GamesU.^2,'b')

figure(38)
plot(Games, V2,'r.')
hold on
p(2,:) = polyfit(GamesU,V2(1:30),2);
plot(GamesU, p(2,3) + p(2,2)*GamesU + p(2,1)*GamesU.^2,'b')

figure(39)
plot(Games, V3,'r.')
hold on
p(3,:) = polyfit(GamesU,V3(1:30),2);
plot(GamesU, p(3,3) + p(3,2)*GamesU + p(3,1)*GamesU.^2,'b')

figure(40)
plot(Games, V4,'r.')
hold on
p(4,:) = polyfit(GamesU,V4(1:30),2);
plot(GamesU, p(4,3) + p(4,2)*GamesU + p(4,1)*GamesU.^2,'b')

figure(41)
plot(Games, V5,'r.')
hold on
p(5,:) = polyfit(GamesU,V5(1:30),2);
plot(GamesU, p(5,3) + p(5,2)*GamesU + p(5,1)*GamesU.^2,'b')

figure(42)
plot(Games, V6,'r.')
hold on
p(6,:) = polyfit(GamesU,V6(1:30),2);
plot(GamesU, p(6,3) + p(6,2)*GamesU + p(6,1)*GamesU.^2,'b')

NPrediction = zeros(length(H),1);
NTTotDiff = zeros(length(H), 1);
NSD = zeros(1,length(GaNu));

% loop that creates new prediciton based on best fit of coefficients
for i=1:length(GaNu)
    if GaNu(i) < 31
        NPrediction(i) = p(1,3) + p(1,2)*GaNu(i) + p(1,1)*GaNu(i).^2 + (p(2,3) + p(2,2)*GaNu(i) + p(2,1)*GaNu(i).^2).*H(i,2) + (p(3,3) + p(3,2)*GaNu(i) + p(3,1)*GaNu(i).^2).*H(i,4) + (p(4,3) + p(4,2)*GaNu(i) + p(4,1)*GaNu(i).^2).*H(i,11) + (p(5,3) + p(5,2)*GaNu(i) + p(5,1)*GaNu(i).^2).*H(i,13) + (p(6,3) + p(6,2)*GaNu(i) + p(6,1)*GaNu(i).^2).*H(i,14);% + (p(7,3) + p(7,2)*H(:,5) + p(7,1)*H(:,5).^2).*(1/abs(H(i,6)-H(i,14)));
    else
        NPrediction(i) = p(1,3) + p(1,2)*30 + p(1,1)*30.^2 + (p(2,3) + p(2,2)*30 + p(2,1)*30.^2).*H(i,2) + (p(3,3) + p(3,2)*30 + p(3,1)*30.^2).*H(i,4) + (p(4,3) + p(4,2)*30 + p(4,1)*30.^2).*H(i,11) + (p(5,3) + p(5,2)*30 + p(5,1)*30.^2).*H(i,13) + (p(6,3) + p(6,2)*30 + p(6,1)*30.^2).*H(i,14);% + (p(7,3) + p(7,2)*H(:,5) + p(7,1)*H(:,5).^2).*(1/abs(H(i,6)-H(i,14)));
    end
end

for j = 1:39
    k = 1;
    for i =1:length(NPrediction)
        if H(i,5) == j
            NDiff(k) = H(i,1) - NPrediction(i);
            Nums(k) = i;
            k = k + 1;
        end
    end

    NTTotDiff(Nums) = NDiff;
    NTTotDiff = NTTotDiff';
    NSD(j) = sqrt(sum((NDiff.^2))/(length(NDiff)-1));
    
    clear -regexp ^NDiff ^Nums
    
end

figure(43)
plot(NPrediction,H(:,1),'g.')

figure(2)
grid on
plot(NSD)

% get performance based on previous game with best fit coefficients
Perform(1) = NaN;
AvPerform(1) = NaN;
for l=2:length(H)
    if (H(l,5) < H(l-1,5)) || (H(l,5) == 1)
        Perform(l) = NaN;
        AvPerform(l) = NaN;
    elseif (H(l,5) == 2 || H(l,5) == 3)
        Perform(l) = NTTotDiff(l-1);
        AvPerform(l) = NaN;
    else
        Perform(l) = NTTotDiff(l-1);
        AvPerform(l) = mean(NTTotDiff(l-3:l-1));
    end
end

TTotPrediction = TTotPrediction';
Perform = Perform';

H(:,18) = Perform;
H(:,19) = AvPerform;
clear -regexp ^Predictors

% total matrix
PT = [H(:,2) H(:,4) H(:,11) H(:,13) H(:,14) H(:,19)];% H(:,16)];% H(:,6)-H(:,14)];
RT = H(:,1);

Predictors = [ones(length(PT),1) PT];
Stats = regress(RT,Predictors); % multivariable regression

lm = fitlm(PT,RT,'linear'); % multivariable regression
%}

%%Spread Prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% allocate space for some variables
SprTTotDiff = zeros(length(H),1);
SprSD = zeros(1, 39);
SprTTotPrediction = zeros(length(H),1);

%%Spread predictor for each game number

avOffPaint = mean(H(:,10));
avDefPaint = mean(H(:,11));

H(:,12) = abs((H(:,10)-avOffPaint) + (H(:,21) - avDefPaint));
H(:,13) = abs((H(:,20)-avOffPaint) + (H(:,11) - avDefPaint));

HCol = [4 9 12 13 15 16 19]; % list of variable columns used for prediction

%create matrix for specific game numbers
for j = 1:1:39
    k=1;
for i=1:1:length(H)
    if H(i,5) == j
       P(k,:) = H(i,HCol);
       R(k,:) = H(i,8);
       Nums(k) = i;
       k = k+1;
    end
end

% set predictor variable so fit function will work
if j < 35
    Predictors = [ones(length(P),1) P]; % final matrix
else
    Predictors = [ones(size(P,1),1) P];
end

Stats = regress(R,Predictors); % stats on matrix
StatTest = regress(R,P); % stats on matrix without bvalue

% set variable coefficients
V1(j) = StatTest(1);
V2(j) = StatTest(2);
V3(j) = StatTest(3);
V4(j) = StatTest(4);
V5(j) = StatTest(5);
V6(j) = StatTest(6);
V7(j) = StatTest(7);

lmSpr = fitlm(P,R,'linear'); % multivariable regression, only uncomment if
%checking stats

SprPrediction = zeros(size(P,1),1);

% loop that calculates predicted spread
for kk=1:length(StatTest)
    SprPrediction = SprPrediction + StatTest(kk).*P(:,kk);% predicted spread 
end

SprDiff = R - SprPrediction; % difference between prediction and actual results

SprTTotDiff(Nums') = SprDiff(:,1);

SprSD(j) = sqrt(sum((SprDiff.^2))/(length(SprDiff)-1)); % standard deviation based on exact coefficients for each game number

SprTTotPrediction(Nums') = SprPrediction(:,1);

clear -regexp ^P ^R ^SprPrediction ^Stats ^StatTest ^SprDiff ^Nums ^RVP ^DifAv ^UsableNums ^AdjUsableNums ^AdjRVP ^AdjDifAv ^AdjPred 

end

% Total fit for all games combined

PT = H(:,HCol);
RT = H(:,8);
Predictors = [ones(length(PT),1) PT];

SFit = fitlm(PT,RT,'linear') %uncomment only if checking stats

Stats = regress(RT,Predictors); % stats on matrix
StatTest = regress(RT,PT);

SpreadPred = zeros(length(PT),1);

% loop that calculates predicted spread
for kk=1:length(StatTest)
    SpreadPred = SpreadPred + StatTest(kk).*PT(:,kk);% predicted spread 
end

SpreadDiff = RT - SpreadPred; % difference between prediction and actual results 

Test = normrnd(0, 11.4, 10619, 1);

figure(4)
plot(RT, SpreadPred,'g.')
title('All Games')
xlabel('RT')
ylabel('Spread Prediction')
grid ON

figure(5)
histogram(SpreadDiff,100,'FaceColor','g')
title('All Games')
grid on

y = 0;

for i=1:length(SprTTotDiff)
    if abs(SprTTotDiff(i)) < 11
        y = y+1;
    end
end

Y = y/length(SprTTotDiff);

figure(7)
histogram(Test,100,'FaceColor','r')
title('Test with 11.4 std')
grid ON

figure(6)
hist(SprTTotDiff, 100)
title('Different variables for Different Game numbers, Exact Coefficients')
grid on

GamesU = (10:30);

% plots for each variable coeffients with functions for best fit
figure(8)
plot(Games, V1,'r.')
hold on
p(1,:) = polyfit(GamesU,V1(GamesU),2);
plot(GamesU, p(1,3) + p(1,2)*GamesU + p(1,1)*GamesU.^2,'b')

figure(9)
plot(Games, V2,'r.')
hold on
p(2,:) = polyfit(GamesU,V2(GamesU),2);
plot(GamesU, p(2,3) + p(2,2)*GamesU + p(2,1)*GamesU.^2,'b')

figure(10)
plot(Games, V3,'r.')
hold on
p(3,:) = polyfit(GamesU,V3(GamesU),2);
plot(GamesU, p(3,3) + p(3,2)*GamesU + p(3,1)*GamesU.^2,'b')

figure(11)
plot(Games, V4,'r.')
hold on
p(4,:) = polyfit(GamesU,V4(GamesU),2);
plot(GamesU, p(4,3) + p(4,2)*GamesU + p(4,1)*GamesU.^2,'b')

figure(12)
plot(Games, V5,'r.')
hold on
p(5,:) = polyfit(GamesU,V5(GamesU),2);
plot(GamesU, p(5,3) + p(5,2)*GamesU + p(5,1)*GamesU.^2,'b')

figure(13)
plot(Games, V6,'r.')
hold on
p(6,:) = polyfit(GamesU,V6(GamesU),2);
plot(GamesU, p(6,3) + p(6,2)*GamesU + p(6,1)*GamesU.^2,'b')

figure(14)
plot(Games, V7,'r.')
hold on
p(7,:) = polyfit(GamesU,V7(GamesU),2);
plot(GamesU, p(7,3) + p(7,2)*GamesU + p(7,1)*GamesU.^2,'b')

NSprPrediction = zeros(length(H),1);
Levels = zeros(length(H),1);

% loop that creates new prediciton based on best fit of coefficients
for i=1:length(GaNu)
    if GaNu(i) > 30
        GaNu(i) = 30;
    end
    C = [GaNu(i)^2 GaNu(i) 1]';
    NSprPrediction(i) = sum((p*C).*H(i,HCol)');
    
    if NSprPrediction(i) > 0
        Levels(i) = 1;
    else
        Levels(i) = -1;
    end
    
end

%allocate space for some variables
NSprSD = zeros(1, 39);
NSprTTotDiff = zeros(length(H),1);

for j = 1:39
k = 1;
    for i =1:length(NSprPrediction)
    if j == H(i,5)
            NSprDiff(k) = H(i,8) - NSprPrediction(i);
            Nums(k) = i;
            k = k+1;
    end
    end
    
    NSprDiff = NSprDiff';
    
    NSprTTotDiff(Nums) = NSprDiff;
    NSprTTotDiff = NSprTTotDiff';
    NSprSD(j) = sqrt(sum((NSprDiff.^2))/(length(NSprDiff)-1));
    
    clear -regexp ^NSprDiff ^Nums
    
end


% get performance based on previous game with best fit coefficients
Perform(1) = NaN;
AvPerform(1) = NaN;
ToPerformCount = 0;
UsePerformCount = 1;
UsePerform = zeros(1,numTeams);
UseAvPerform = zeros(1,numTeams);
UseOPaint = zeros(1,numTeams);
UseDPaint = zeros(1,numTeams);
Tester = zeros(1,numTeams);
AvPerform2(1) = 1;
TeamCount = 1;

for l=2:length(H)
    if (H(l,5) <= H(l-1,5)) || (H(l,5) == 1)
        Perform(l) = NaN;
        AvPerform(l) = NaN;
                TeamCount = TeamCount + 1;
        AvPerform2(l) = TeamCount;
    elseif (H(l,5) == 2 || H(l,5) == 3)
        Perform(l) = NSprTTotDiff(l-1);
        AvPerform(l) = NaN;
                AvPerform2(l) = TeamCount;
    else
        Perform(l) = NSprTTotDiff(l-1);
        AvPerform(l) = mean(NSprTTotDiff(l-3:l-1));
                AvPerform2(l) = TeamCount;
        ToPerformCount = ToPerformCount + 1;
               Tester(ToPerformCount) = H(l,5);
    end
    
    if ((H(l,5) < H(l-1,5)) && (l>3))
        UsePerform(UsePerformCount) = Perform(l-1);
        UseAvPerform(UsePerformCount) = AvPerform(l-1);
        UseOPaint(UsePerformCount) = H(l-1,10)-avOffPaint;
        UseDPaint(UsePerformCount) = H(l-1,11)-avDefPaint;
        UsePerformCount = UsePerformCount + 1;
    end
    if l==length(H)
        UsePerform(UsePerformCount) = Perform(l);
        UseAvPerform(UsePerformCount) = AvPerform(l);
        UseOPaint(UsePerformCount) = H(l,10)-avOffPaint;
        UseDPaint(UsePerformCount) = H(l,11)-avDefPaint;
    end
end

maxtest = min(Tester)

TTotPrediction = TTotPrediction';
ToPerform = Perform';
V1 = zeros(1,Games(end));
V2 = zeros(1,Games(end));
V3 = zeros(1,Games(end));
V4 = zeros(1,Games(end));
V5 = zeros(1,Games(end));
V6 = zeros(1,Games(end));
V7 = zeros(1,Games(end));
V8 = zeros(1,Games(end));
V9 = zeros(1,Games(end));
V10 = zeros(1,Games(end));

% Set columns for performance
numToPerform = 22;
numAvPerform = 23;
numLevels = 24;
H(:,numToPerform) = ToPerform;
H(:,numAvPerform) = AvPerform;
H(:,numLevels) = Levels;
HCol = [HCol numToPerform numAvPerform];

for j = 1:1:39
    k=1;
for i=1:1:length(H)
    if H(i,5) == j
       P(k,:) = H(i,HCol);
       R(k,:) = H(i,8);
       Nums(k) = i;
       k = k+1;
    end
end

if j < 37
    Predictors = [ones(length(P),1) P]; % final matrix
else
    Predictors = [ones(size(P,1),1) P];
end

Stats = regress(R,Predictors); 
StatTest = regress(R,P);


% set variable coefficients
V1(j) = StatTest(1);
V2(j) = StatTest(2);
V3(j) = StatTest(3);
V4(j) = StatTest(4);
V5(j) = StatTest(5);
V6(j) = StatTest(6);
V7(j) = StatTest(7);
V8(j) = StatTest(8);
V9(j) = StatTest(9);
% V10(j) = StatTest(10);

lmPerform = fitlm(P,R,'linear')

SprAdjPrediction = zeros(size(P,1),1);

% loop that calculates predicted spread
for kk=1:length(StatTest)
    SprAdjPrediction = SprAdjPrediction + StatTest(kk).*P(:,kk);% predicted spread 
end

SprAdjDiff = SprAdjPrediction - R; % difference between prediction and actual results

clear -regexp ^P ^R ^SprAdjPrediction ^Stats ^SprAdjDiff ^Nums

end

figure(15)
plot(Games, V1,'r.')
hold on
p(1,:) = polyfit(GamesU,V1(GamesU),2);
plot(GamesU, p(1,3) + p(1,2)*GamesU + p(1,1)*GamesU.^2,'b')

figure(16)
plot(Games, V2,'r.')
hold on
p(2,:) = polyfit(GamesU,V2(GamesU),2);
plot(GamesU, p(2,3) + p(2,2)*GamesU + p(2,1)*GamesU.^2,'b')

figure(17)
plot(Games, V3,'r.')
hold on
p(3,:) = polyfit(GamesU,V3(GamesU),2);
plot(GamesU, p(3,3) + p(3,2)*GamesU + p(3,1)*GamesU.^2,'b')

figure(18)
plot(Games, V4,'r.')
hold on
p(4,:) = polyfit(GamesU,V4(GamesU),2);
plot(GamesU, p(4,3) + p(4,2)*GamesU + p(4,1)*GamesU.^2,'b')

figure(19)
plot(Games, V5,'r.')
hold on
p(5,:) = polyfit(GamesU,V5(GamesU),2);
plot(GamesU, p(5,3) + p(5,2)*GamesU + p(5,1)*GamesU.^2,'b')

figure(20)
plot(Games, V6,'r.')
hold on
p(6,:) = polyfit(GamesU,V6(GamesU),2);
plot(GamesU, p(6,3) + p(6,2)*GamesU + p(6,1)*GamesU.^2,'b')

figure(21)
plot(Games, V7,'r.')
hold on
p(7,:) = polyfit(GamesU,V7(GamesU),2);
plot(GamesU, p(7,3) + p(7,2)*GamesU + p(7,1)*GamesU.^2,'b')

figure(22)
plot(Games, V8,'r.')
hold on
p(8,:) = polyfit(GamesU,V8(GamesU),2);
plot(GamesU, p(8,3) + p(8,2)*GamesU + p(8,1)*GamesU.^2,'b')

figure(23)
plot(Games, V9,'r.')
hold on
p(9,:) = polyfit(GamesU,V9(GamesU),2);
plot(GamesU, p(9,3) + p(9,2)*GamesU + p(9,1)*GamesU.^2,'b')

% figure(24)
% plot(Games, V10,'r.')
% hold on
% p(10,:) = polyfit(GamesU,V10(GamesU),2);
% plot(GamesU, p(10,3) + p(10,2)*GamesU + p(10,1)*GamesU.^2,'b')

% Get Final Predictions With Final Coefficients

FinSprPrediction = zeros(length(H),1);

% loop that creates new prediciton based on best fit of coefficients
for i=1:length(GaNu)
    if GaNu(i) > 30
        GaNu(i) = 30;
    end
    C = [GaNu(i)^2 GaNu(i) 1]';
    FinSprPrediction(i) = sum((p*C).*H(i,HCol)');
end

%allocate space for some variables
FinSprTTotDiff = zeros(length(H),1);
FinSprSD = zeros(1, 39);

for j = 1:39
k = 1;
    for i =1:length(FinSprPrediction)
    if j == H(i,5)
            FinSprDiff(k) = H(i,8) - FinSprPrediction(i);
            Nums(k) = i;
            k = k+1;
    end
    end
    
    FinSprDiff = FinSprDiff';

    FinSprTTotDiff(Nums) = FinSprDiff;
    FinSprTTotDiff = FinSprTTotDiff';
    FinSprSD(j) = sqrt(sum((FinSprDiff.^2))/(length(FinSprDiff)-1));
    
    clear -regexp ^FinSprDiff ^Nums
    
end

figure(30)
hist(FinSprTTotDiff, 100)
title('Different variables for Different Game numbers, Best fit coefficients, performance')
grid on

PT2 = H(:,HCol);
RT2 = H(:,8);
Predictors = [ones(length(PT2),1) PT2];

SFit2 = fitlm(PT2,RT2,'linear')

z = 0;

for i=1:length(FinSprTTotDiff)
    if abs(FinSprTTotDiff(i)) < 1
        z = z+1;
    end
end

% rowcount = 1;
% for ii=1:length(H)
%     if H(ii,5) > 10
%         Hnew(rowcount,:) = H(ii,:);
%         %Rnew(rowcount,:) = R(ii,:);
%         rowcount = rowcount + 1;
%     end
% end
%         
% PTnew = Hnew(:,HCol);
% RTnew = Hnew(:,8);
% 
% SFit3 = fitlm(PTnew,RTnew,'linear')

Z = z/ToPerformCount;

%%Check Standard Deviation by Spread Prediction

RoundDif = round(FinSprTTotDiff);

RoundSpr = round(FinSprPrediction);

STDbySPRSum = zeros(1,37);
STDbySPRcount = zeros(1,37);
STDbySPR = zeros(1,37);

figure(36)
hist(FinSprTTotDiff,100)
title('Spread Difference')
grid on

figure(25)
hist(FinSprPrediction,100)
title('Spread Prediction')
grid on

x = -60:60;

idxValid = ~isnan(FinSprPrediction);
poly = polyfit(H(idxValid,8),FinSprPrediction(idxValid),1);

figure(26)
plot(H(:,8),FinSprPrediction,'g.')
xlabel('Actual Spread')
ylabel('Predicted Spread')
hold on
plot(x,x,'b')
hold on
plot(x,poly(1)*x + poly(2),'r')
grid on

figure(27)
hist(H(:,8),150)
title('Actual Spread')
grid on

figure(28)
plot(STDbySPR,'g.')
grid on

%%show matrix of gamerun stats

UseStats = [UseSpread' UseSOS' UsePerform(1:numTeams)' UseAvPerform(1:numTeams)' UseOPaint(1:numTeams)' UseDPaint(1:numTeams)'];

% fileID = fopen('TStats.txt','r');
% TUseStats = textscan(fileID,'%f %f %f %f %f %f','Delimiter','\t'); % extract stats from team's txtfile
% fclose(fileID);
% 
% for i =1:1:6 %% create individual cell vectors
%     UseStats(:,i)=TUseStats{i};
% end 
% 
% fileID = fopen('GameN.txt','r');
% TGameN = textscan(fileID,'%f','Delimiter','\t'); % extract stats from team's txtfile
% fclose(fileID);
% 
% for i =1:1:1 %% create individual cell vectors
%     UseGameN(:,i)=TGameN{i};
% end 

end