function [H, UseStats, p, FinSprSD, UseGameN, SendLocSpread] = Tester2()

clear
clc
close all

format long
fileID = fopen('TeamNamesBB.txt','r');
TeamNames = textscan(fileID,'%s','Delimiter','\n'); % extract stats from team's txtfile
fclose(fileID);

year = '2018';

count = 0;
numTeams = 351;
UseSOS = zeros(1,numTeams);
UseSpread = zeros(1,numTeams);
UseGameN = zeros(1,numTeams);
UseScore = zeros(1,numTeams);
ToWin = zeros(1,numTeams);
LastWin = zeros(1,numTeams);
ToWinPct = zeros(1,numTeams);
LastWinPct = zeros(1,numTeams);
UseLoc = zeros(1,numTeams);
LastSpread = zeros(1,numTeams);
LastTapeSpread = zeros(1,numTeams);
LastLoc = zeros(1,numTeams);
UseLastGame = zeros(1,numTeams);
ToPassTapeAvSpread = zeros(1,numTeams);
TheSOS = zeros(1,numTeams);
TotTest2 = zeros(1,numTeams);
TaperedSOS = zeros(1,numTeams);
LastTapeAvSOSnum = zeros(1,numTeams);
UseOpponent = zeros(1,numTeams);
TotTest = zeros(351,1);

SendLocSpread = zeros(numTeams,3);

QUseStats = load('QStats.mat');
QUseSpread = QUseStats.Spread; %mean spread of all games for each team based off excel data
UseGameN = QUseStats.GameN; % total number of games played for each team based off excel data

TotStats = zeros(10800,25);

for h=1:1:numTeams %% loop that goes through all teams to collect stats and opp stats
  
    Name = [TeamNames{1}{h},'BBXS',year,'.mat']; % name of team collecting data for
    
    [ExcTest] = load(Name); % this file is saved when importing from excel
        
        %determine teams season info
        A(:,1) = ExcTest.ExcNum(:,1); % team score
        A(:,2) = ExcTest.ExcNum(:,2); % opponent score
        A(:,4) = ExcTest.ExcNum(:,6); % team three pointers made
        
        if (size(char(ExcTest.ExcText(:,1)),2)) > 1
            A(:,3) = ' ';
            D = char(ExcTest.ExcText(:,1));% Opponent's opponents
        else
            A(:,3) = char(ExcTest.ExcText(:,1)); % location
            D = char(ExcTest.ExcText(:,2));% Opponent's opponents
        end
        
        S = zeros(size(A,1),1);
        OS = zeros(size(A,1),1);
        
        %%This section deteremines the number of days between games
%         Date = char(ExcTest.ExcText(:,end));
%         
%         Break = zeros(length(Date)-1,1);
%         Month = zeros(length(Date)-1,1);
%         Day = zeros(length(Date)-1,1);
%         
%         Break(1) = 3;
%         
%         for xx=1:length(Date)-1
%             
%             if Date(xx,2) == '/'
%                 
%                 Month(xx) = str2num(Date(xx,1));
%                 
%                 if Date(xx,4) == '/'
%                     
%                     Day(xx) = str2num(Date(xx,3));
%                     
%                 else
%                     
%                     Day(xx) = str2num(Date(xx,3:4));
%                     
%                 end
%                 
%             else
%                 
%                 Month(xx) = str2num(Date(xx,1:2));
%                 
%                 if Date(xx,5) == '/'
%                     
%                     Day(xx) = str2num(Date(xx,4));
%                     
%                 else
%                     
%                     Day(xx) = str2num(Date(xx,4:5));
%                     
%                 end
%                 
%             end
%             
%             if xx >= 2
%                 
%                 if Month(xx) == Month(xx-1)
%                     
%                     Break(xx) = Day(xx) - Day(xx-1);
%                     
%                 elseif Month(xx) == 1 || Month(xx) == 2 || Month(xx) == 4
%                     
%                     Break(xx) = Day(xx) + 31 - Day(xx-1);
%                     
%                 elseif Month(xx) == 12
%                     
%                     Break(xx) = Day(xx) + 30 - Day(xx-1);
%                     
%                 else 
%                     
%                     Break(xx) = Day(xx) + 28 - Day(xx-1);
%                     
%                 end
%                 
%             end
%             
%         end
%         
%         LastBreak(h) = Break(end);

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
        SOSnumTapeTest = zeros(1,length(S));
        Pct3 = zeros(1,length(Snew)); % allocate space for percent of points from 3s
        
        TapeAvSpread = zeros(1,length(S));
        TapeAvSpread(1) = Spread(1); %first tapered spread is just first spread
        
        for gg=2:length(S)
            
            TapeTest = zeros(1,gg);
            
            for vv=1:1:gg

                TapeTest(vv) = Spread(vv)*(.5 + ((1/(gg-1))*.5)*(vv-1));

            end

            TapeAvSpread(gg) = mean(TapeTest);
        
        end
        
%        [TapeAvSpread' Spread]
        
        PassTapeMean = TapeAvSpread(end);
        
        ToPassTapeAvSpread(h) = PassTapeMean;
        
        TapeAvSpread = TapeAvSpread(1:end-1);
        
        HomeSpreadCount = 1;
        HomeSpread = 3;
        AwaySpreadCount = 1;
        AwaySpread = -3;
        NeutralSpreadCount = 1;
        NeutralSpread = 0;
        LocSpread = zeros(1,length(Snew));
        
        Win = zeros(1,length(Snew));
        WinPct = zeros(1,length(Snew));
        
        for i =1:1:length(Snew) % determines average team and opp score based on history before that game
            TotPoints = sum(S(1:i));
            TotOPoints = sum(OS(1:i));
            SThrees = sum(Threes(1:i));
            SOThrees = sum(OThrees(1:i));
            PctPaint(i) = (TotPoints - 3*SThrees)/TotPoints;
            PctOPaint(i) = (TotOPoints - 3*SOThrees)/TotOPoints;
            avSc(i) = mean(S(1:i));
            avSpread(i) = mean(Spread(1:i)); % average spread based on games up to and after game "i"   
            avOSc(i) = mean(OS(1:i));
            TotAvSc(i) = mean(S(1:length(Snew)));
            Pct3(i) = (mean(Threes(1:i))*3)/avSc(i);
            
            if S(i) > OS(i)
                
                Win(i) = 1;
                
            else
                
                Win(i) = 0;
                
            end
            
            WinPct(i) = sum(Win)/i;
            
            if S(end) > OS(end)
                
                ToWin(h) = 1;
                
            else
                
                ToWin(h) = 0;
                
            end
            
            ToWinPct(h) = (sum(Win) + ToWin(h))/length(S);
            
            if Loc(i) == ' '
                                    
                HomeSpread(HomeSpreadCount) = S(i) - OS(i);
                
                HomeSpreadCount = HomeSpreadCount + 1;
                
            elseif Loc(i) == 'N'
                
                NeutralSpread(NeutralSpreadCount) = S(i) - OS(i);
                
                NeutralSpreadCount = NeutralSpreadCount + 1;
                
            else
                
                AwaySpread(AwaySpreadCount) = S(i) - OS(i);
                
                AwaySpreadCount = AwaySpreadCount + 1;
                
            end
            
            % determine location
            if Loc(i+1) == ' '
                
                Loc(i) = 1;
                
                if HomeSpreadCount > 1
                
                    LocSpread(i) = mean(HomeSpread);
                    
                else
                    
                    LocSpread(i) = avSpread(i);
                    
                end
                
            elseif Loc(i+1) == 'N'
                
               Loc(i) = 0;
               
               if HomeSpreadCount > 1 && AwaySpreadCount > 1
                   
                   LocSpread(i) = mean([mean(AwaySpread) mean(HomeSpread)]);
                   
               else
                   
                   LocSpread(i) = avSpread(i);
                   
               end
                   
            else
                
                Loc(i) = -1;
                
                if AwaySpreadCount > 1
                
                    LocSpread(i) = mean(AwaySpread);
                    
                else
                    
                    LocSpread(i) = avSpread(i);
                    
                end
                
            end
                   
        end
        
        LastWin(h) = Win(end);
            
        LastWinPct(h) = WinPct(end);
        
        LastLoc(h) = LocSpread(end);
        
        if Loc(end) == ' '
            
            UseLoc(h) = 1;
            
            HomeSpread(HomeSpreadCount) = S(end) - OS(end);
            
        elseif Loc(end) == 'N'
            
               UseLoc(h) = 0;
               
               NeutralSpread(NeutralSpreadCount) = S(end) - OS(end);
               
        else
            
            UseLoc(h) = -1;
            
            AwaySpread(AwaySpreadCount) = S(end) - OS(end);
            
        end
    
        SpreadMat = [mean(HomeSpread) mean([mean(HomeSpread) mean(AwaySpread)]) mean(AwaySpread)];
        SendLocSpread(h,1:3) = SpreadMat;

        %%extra stats for use
        UseScore(h) = S(end);
        LastSpread(h) = mean(Spread(1:end-1));
        LastTapeSpread(h) = TapeAvSpread(end);
        UseLastGame(h) = Spread(end);
        
        %%useful team stats      
        newStats = [Snew avSc' avOSc' Loc(1:(end-1)) UsedGames' WinPct' Win' SpreadN avSpread' PctPaint' PctOPaint' TapeAvSpread' LocSpread'];
     
  %///////////////////////// only if new excel data
  
  SOSnum = zeros(1,length(S));
  
  newOppStats = zeros(length(Snew),12);
  
  for j =1:length(S)
  
 %/////////////////// Only use if using for analysis
 
        if j > 1
        
        Dnew(1,:) = deblank(D(j,1:1:end)); % get Game Opp name from getting rid of trailing blank space
 
   Name = [Dnew,'BBXS',year,'.mat'];
    
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
        OSpread = S2 - OS2;
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
                OPctPaint = 0;
                OPctOPaint = 0;
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
        
        OTapeTest(1) = OSpread(1); %first tapered spread is just first spread
        
        if k > 1
            
            OTapeTest = zeros(1,k-1);
            
            for vv=1:1:k-1

                OTapeTest(vv) = OSpread(vv)*(.5 + ((1/(k-2))*.5)*(vv-1));

            end
        
        end

        OTapeAvSpread = mean(OTapeTest);
        
        if k ==2
            
            OTapeAvSpread = OSpread(1);
            
        end
        
        PointDif = zeros(1,length(S)-1);
        
       % Previous Team Opponents 
            for m=1:1:(j-1)           
                Fnew(1,:) = deblank(D(m,1:1:end));
                
                Name = [Fnew,'BBXS',year,'.mat'];
    
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
            
            UseSOS(h) = mean([PointDif avSpread2]); % not used
            
            OPointDif = zeros(1,length(OGU));
            if OGU == 0
                OPointDif = 0;
            else
            
            for n=1:1:OGU
            Gnew(1,:) = deblank(D2(n,1:1:end)); % get Game Opp name from getting rid of trailing blank space

                Name = [Gnew,'BBXS',year,'.mat'];
    
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
            
            TapeTestSOS(1) = 0;%%%%%%%
             
            if j < 3
            
                TapeAvSOS = SOS; %first tapered spread is just first spread
                
            else
                
                TapeTestSOS = zeros(1,j-1);
                
               for vv=1:1:j-1

                    TapeTestSOS(vv) = PointDif(vv)*(.5 + ((1/(j-2))*.5)*(vv-1));

               end

               TapeAvSOS = mean(TapeTestSOS);
            
            end
                        
        %%useful opposing teams stats
        newOppStats(j-1,:) = [AvOS AvOOS TotOOS SOS OSOS OGU OPct3 avSpread2 OPctPaint OPctOPaint OTapeAvSpread TapeAvSOS]; 
        
%                   [newOppStats(:,4) newOppStats(:,11) PointDif(1:j-1)' TapeTestSOS']
% [newOppStats(:,8) newOppStats(:,11)]
% OTapeTest'
        
        end
        
        clear -regexp ^Dnew ^ExcTest2 ^A2 ^S2 ^OS2 ^GameNum ^PointDif ^OPointDif ^AvOS ^AvOOS
        
        %////////////////////////////////////////////////
 
            for rr = 1:numTeams
          
                QString(1,:) = deblank(D(j,1:1:end));
            
                if (length(QString) == length(TeamNames{1}{rr})) & (QString == TeamNames{1}{rr})
              
                    SOSnum(j) = QUseSpread(rr);
                    
                break;
              
                end
                
            end
            
            clear -regexp ^QString  ^TapePointDif
            
          % combine stats once all data is entered
  end

    for gg=1:length(S)

        SOSnumTapeTest(gg) = SOSnum(gg)*(.5 + ((1/(length(S)-1))*.5)*(gg-1));

    end
    
    TaperedSOS(h) = mean(SOSnumTapeTest);
    
    LastSOSnumTapeTest = zeros(1,length(Snew));
    
    for gg=1:length(Snew)

        LastSOSnumTapeTest(gg) = SOSnum(gg)*(.5 + ((1/(length(Snew)-1))*.5)*(gg-1));

    end
    
    LastTapeAvSOSnum(h) = mean(LastSOSnumTapeTest);
        
            TotStats(count+1:count+length(avSc),:) = [newStats newOppStats];
            count = count + length(avSc);
%             
%             [SOSnum' SOSnumTapeTest']

  
  UseOpponent(h) = rr;
  
  TheSOS(h) = mean(SOSnum(1:end-1));
  TotTest2(h) = mean(SOSnum);
  
%   LastTapeSOS(h) = TapeAvSOS;
  
%   SOSnum(end);
  
  % SOSnum are final strengths of schedules
  % TapeSOS is a running strength of schedule throughout season
  
%   HomeSpread'
%   NeutralSpread'
%   AwaySpread'
%   LocSpread'
  
  clear -regexp ^Loc ^A ^ExcTest ^newStats ^Snew ^Spread ^GameNum ^PctPaint ^PctOPaint ^SOSnum ^TapeAvSpread ^TapeTest ^SOS

  h

end

%///////////////////////////// Only use when importing new stats for
%analysis

UTotStats = TotStats(1,:);
% get rid of '0' scores - put UTotStats into separtate file
for i =2:1:length(TotStats)
    if TotStats(i,14) ~= 0
        UTotStats = [UTotStats;TotStats(i,:)];
    end
end

USName = 'UTotStats.mat';
save(USName, 'UTotStats');
 
%//////////////////////// 

    ULoad = load('UTotStats.mat');
    
    UTotStats = ULoad.UTotStats;

%//////////////////////////////////////////////////////////

%% Statistical Analysis

% close all
% clc
% clear
x=1:1:150;
Games = 1:1:34;    
numStats = 25;
numTeams = 351;

% using avgscore, location, both sos, and opp opp's avgscore for games only before game played

fileID = fopen('TotStats2016.txt','r');
TT = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter','\t'); % extract stats from team's txtfile
fclose(fileID);

for i =1:1:numStats %% create individual cell vectors
    H(:,i)=TT{i};
end

fileID = fopen('TotStats2017.txt','r');
TT2 = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter','\t'); % extract stats from team's txtfile
fclose(fileID);

for i =1:1:numStats %% create individual cell vectors
    H2(:,i)=TT2{i};
end

fileID = fopen('2018Temp.txt','r');
TT3 = textscan(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','Delimiter','\t'); % extract stats from team's txtfile
fclose(fileID);

for i =1:1:numStats %% create individual cell vectors
    H3(:,i)=TT3{i};
end

%//////////////////////// Change H3 to UTotStats if importing
H = [UTotStats; H; H2];%[H3; H; H2];

GaNu = H(:,5); % game numbers
Perform = zeros(length(H),1);
AvPerform = zeros(length(H),1);
AvPerform2 = zeros(length(H),1);
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

% H(:,12) = abs((H(:,10)-avOffPaint) + (H(:,21) - avDefPaint));
% H(:,13) = abs((H(:,20)-avOffPaint) + (H(:,11) - avDefPaint));

HCol = [4 9 24 17 18 21 13]; % list of variable columns used for prediction, Loc-Spread-SOS-OSOS-OppSpread-OTapeSpread

H10count = 1;

for yy=1:length(H)

    if H(yy,5) >=10
        
        HR10(H10count) = H(yy,8);
        
        HP10(H10count,:) = H(yy, HCol);
        
        H10count = H10count + 1;
        
    end
    
end

%create matrix for specific game numbers
for j = 1:1:34
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
    Predictors = [ones(size(P,1),1) P]; % final matrix
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
% V8(j) = StatTest(8);

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

SFit10 = fitlm(HP10,HR10,'linear') %fit for games over 10

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

% figure(15)
% plot(Games, V8,'r.')
% hold on
% p(8,:) = polyfit(GamesU,V8(GamesU),2);
% plot(GamesU, p(8,3) + p(8,2)*GamesU + p(8,1)*GamesU.^2,'b')


% 
% NSprPrediction = zeros(length(H),1);
% Levels = zeros(length(H),1);

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

for j = 1:34
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
UsePerformTest = zeros(1,numTeams);
UseAveragePerform = zeros(1,numTeams);
UseAv2Perform = zeros(1,numTeams);
Pred = zeros(1,numTeams);
AvPerform2(1) = NaN;
TeamCount = 1;

for l=2:length(H)
    if (H(l,5) <= H(l-1,5)) || (H(l,5) == 1)
        Perform(l) = NaN;
        AvPerform(l) = NaN;
        AvPerform2(l) = NaN;

    elseif (H(l,5) == 2 || H(l,5) == 3)
        Perform(l) = NSprTTotDiff(l-1);
        AvPerform(l) = NaN;
        AvPerform2(l) = NaN;
        
        if H(l,5) == 3
            
            AvPerform2(l) = mean(NSprTTotDiff(l-2:l-1));
            
        end

    else
        Perform(l) = NSprTTotDiff(l-1);
        AvPerform(l) = mean(NSprTTotDiff(l-3:l-1));
        AvPerform2(l) = mean(NSprTTotDiff(l-2:l-1));
%         ToPerformCount = ToPerformCount + 1;
    end
    
    if (H(l,5) < H(l-1,5))
        UsePerform(UsePerformCount) = NSprTTotDiff(l-1);
        UseAvPerform(UsePerformCount) = mean(NSprTTotDiff(l-3:l-1));
        UseAv2Perform(UsePerformCount) = mean(NSprTTotDiff(l-2:l-1));
        UsePerformCount = UsePerformCount + 1;
    end
    if l==length(H)
        UsePerform(UsePerformCount) = Perform(l);
        UseAvPerform(UsePerformCount) = AvPerform(l);
        UseAv2Perform(UsePerformCount) = AvPerform2(l);
    end
end

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
% V9 = zeros(1,Games(end));
% V10 = zeros(1,Games(end));

% Set columns for performance
numAvPerform = 26;
numLevels = 27;
H(:,numAvPerform) = AvPerform;
H(:,numLevels) = Levels;
HCol = [HCol numAvPerform];

for j = 1:1:34
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
    Predictors = [ones(size(P,1),1) P]; % final matrix
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
% V9(j) = StatTest(9);
% V10(j) = StatTest(10);

lmPerform = fitlm(P,R,'linear');

SprAdjPrediction = zeros(size(P,1),1);

% loop that calculates predicted spread
for kk=1:length(StatTest)
    SprAdjPrediction = SprAdjPrediction + StatTest(kk).*P(:,kk);% predicted spread 
end

SprAdjDiff = SprAdjPrediction - R; % difference between prediction and actual results

clear -regexp ^P ^R ^SprAdjPrediction ^Stats ^SprAdjDiff ^Nums

end

figure(16)
plot(Games, V1,'r.')
hold on
p(1,:) = polyfit(GamesU,V1(GamesU),2);
plot(GamesU, p(1,3) + p(1,2)*GamesU + p(1,1)*GamesU.^2,'b')

figure(17)
plot(Games, V2,'r.')
hold on
p(2,:) = polyfit(GamesU,V2(GamesU),2);
plot(GamesU, p(2,3) + p(2,2)*GamesU + p(2,1)*GamesU.^2,'b')

figure(18)
plot(Games, V3,'r.')
hold on
p(3,:) = polyfit(GamesU,V3(GamesU),2);
plot(GamesU, p(3,3) + p(3,2)*GamesU + p(3,1)*GamesU.^2,'b')

figure(19)
plot(Games, V4,'r.')
hold on
p(4,:) = polyfit(GamesU,V4(GamesU),2);
plot(GamesU, p(4,3) + p(4,2)*GamesU + p(4,1)*GamesU.^2,'b')

figure(20)
plot(Games, V5,'r.')
hold on
p(5,:) = polyfit(GamesU,V5(GamesU),2);
plot(GamesU, p(5,3) + p(5,2)*GamesU + p(5,1)*GamesU.^2,'b')

figure(21)
plot(Games, V6,'r.')
hold on
p(6,:) = polyfit(GamesU,V6(GamesU),2);
plot(GamesU, p(6,3) + p(6,2)*GamesU + p(6,1)*GamesU.^2,'b')

figure(22)
plot(Games, V7,'r.')
hold on
p(7,:) = polyfit(GamesU,V7(GamesU),2);
plot(GamesU, p(7,3) + p(7,2)*GamesU + p(7,1)*GamesU.^2,'b')

figure(23)
plot(Games, V8,'r.')
hold on
p(8,:) = polyfit(GamesU,V8(GamesU),2);
plot(GamesU, p(8,3) + p(8,2)*GamesU + p(8,1)*GamesU.^2,'b')

% figure(24)
% plot(Games, V9,'r.')
% hold on
% p(9,:) = polyfit(GamesU,V9(GamesU),2);
% plot(GamesU, p(9,3) + p(9,2)*GamesU + p(9,1)*GamesU.^2,'b')

% figure(24)
% plot(Games, V10,'r.')
% hold on
% p(10,:) = polyfit(GamesU,V10(GamesU),2);
% plot(GamesU, p(10,3) + p(10,2)*GamesU + p(10,1)*GamesU.^2,'b')

for bb=1:numTeams
    if UseGameN(bb) > 30
        UseGameN(bb) = 30;
    end
    
   D = [UseGameN(bb)^2 UseGameN(bb) 1]';
   Pred(bb) = sum((p*D).*[UseLoc(bb) LastSpread(bb) LastTapeSpread(UseOpponent(bb)) TheSOS(bb) TheSOS(UseOpponent(bb)) LastSpread(UseOpponent(bb)) LastLoc(bb) UseAvPerform(bb)]');%LastTapeAvSOSnum(bb)   
   
   UsePerformTest(bb) = UseLastGame(bb) - Pred(bb);
   UseAveragePerform(bb) = (UseAv2Perform(bb)*2 + UsePerformTest(bb))/3;
   
end


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

for j = 1:34
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

H10count = 1;

for zz=1:length(H)

    if H(zz,5) >=10
        
        HR10(H10count) = H(zz,8);
        
        HP10(H10count,1:length(HCol)) = H(zz, HCol);
        
        H10count = H10count + 1;
        
    end
    
end

SFit210 = fitlm(HP10,HR10,'linear')

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

%//////////////Use if new excel data
UseStats = [ToWinPct' ToWin' QUseSpread ToPassTapeAvSpread' TotTest2' TaperedSOS' UseAveragePerform'];

%//////////////////////// only save if new excel data
%  TName = 'TStats.mat';
%      
%     save(TName, 'UseStats');
%     
%     LookStats = load('TStats.mat');
%     
%     UseStats = LookStats.UseStats;
    
end