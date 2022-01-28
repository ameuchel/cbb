clear
clc
close all

fileID = fopen('TeamNamesBB.txt','r');
TeamNames = textscan(fileID,'%s','Delimiter','\n'); % extract stats from team's txtfile
fclose(fileID);

numTeams = 351;
%TeamRankings = zeros(1,351);
TeamRankings(1) = 1;
TeamNameRankings = strings(1,1);
TeamNameRankings(1) = TeamNames{1}{1};

%////////////////////// % Use if importing new games
%Exc2Mat2();


MatchUp = Matchup();

[StatMat, UseStats, Coeff, SD, GameN, SendLocSpread] = Tester2();

close all

% for tt=2:numTeams
%     
% for ss=1:tt-1    
% 
%     if tt == TeamRankings(ss)
%     else
%     MatchUp = [tt  TeamRankings(ss) 0;
%                TeamRankings(ss) tt 0];

Sum = zeros(size(MatchUp,1),1);
Prediction = zeros(size(MatchUp,1)/2,1);
Z = zeros(size(MatchUp,1),1);
Z2 = zeros(size(MatchUp,1)/2,1);
pval = zeros(size(MatchUp,1)/2,1);
pval2 = zeros(size(MatchUp,1)/2,1);
Net = zeros(size(MatchUp,1)/2,1);
Net2 = zeros(size(MatchUp,1)/2,1);
Choose = zeros(size(MatchUp,1)/2,1);
Choose2 = zeros(size(MatchUp,1)/2,1);
OD = zeros(size(MatchUp,1)/2,1);
SpVal = 110/210;
ZS = 0.0597170998379;

% Prediction is what the home team is going to win or lose by
% First column is spread

for jj =1:size(MatchUp,1)
    
    if MatchUp(jj,3) == 1
        
        LocateSpread = SendLocSpread(MatchUp(jj,1),1);
        
    elseif MatchUp(jj,3) == -1
        
        LocateSpread = SendLocSpread(MatchUp(jj,1),3);
        
    else
        
        LocateSpread = SendLocSpread(MatchUp(jj,1),2);
        
    end
    
    Vars = [MatchUp(jj,3) UseStats(MatchUp(jj,1),3) UseStats(MatchUp(jj,2),4) UseStats(MatchUp(jj,1),5) UseStats(MatchUp(jj,2),5) UseStats(MatchUp(jj,2),3) LocateSpread UseStats(MatchUp(jj,1),7)];%UseStats(MatchUp(jj,1),4)   
    if GameN(MatchUp(jj,1)) > 30
        GameN(MatchUp(jj,1)) = 30;
    end
for ii=1:length(Coeff)
    Part = (Coeff(ii,3) + Coeff(ii,2)*GameN(MatchUp(jj,1)) + Coeff(ii,1)*GameN(MatchUp(jj,1))^2)*Vars(ii);%GameN(MatchUp(jj,1)),(Coeff(ii,3) + Coeff(ii,2)*pp + Coeff(ii,1)*pp^2)
    Sum(jj) = Sum(jj) + Part;
end
end

for kk=1:size(MatchUp,1)
    if mod(kk,2) == 0
        Prediction(kk/2) = (Sum(kk-1)-Sum(kk))/2;
        
        Z2(kk/2) = (Prediction(kk/2))/SD(round(mean([GameN(MatchUp(kk)) GameN(MatchUp(kk-1))])));
        pval2(kk/2) = normcdf(Z2(kk/2));

        HomeSpread(kk/2) = (-ZS*SD(round(mean([GameN(MatchUp(kk)) GameN(MatchUp(kk-1))]))))+Prediction(kk/2);

        AwaySpread(kk/2) = Prediction(kk/2) + (Prediction(kk/2) - HomeSpread(kk/2));
        
        for mm=1:2
            if mm == 2
                pval2(kk/2) = 1 - pval2(kk/2);
            end
            if pval2(kk/2) > 0.5
                OD(kk/2,mm) = -100*pval2(kk/2)/(1-pval2(kk/2));
            else
                OD(kk/2,mm) = 100*(1-pval2(kk/2))/pval2(kk/2);
            end
            PctOdds(kk) = 1 - (100/abs(OD(kk/2,mm)));
        end
        
    end
end

% if Prediction > 0
%     if ss == 1
%         TeamRankings = [tt TeamRankings];
%         TeamNameRankings = [TeamNames{1}{tt}; TeamNameRankings];
%     else
%     %nTeamRankings(ss) = tt;
%     TeamRankings = [TeamRankings(1:ss-1) tt TeamRankings(ss:end)];
%     TeamNameRankings = [TeamNameRankings(1:ss-1); TeamNames{1}{tt}; TeamNameRankings(ss:end)];
%     end
%      break;
% else
%      if ss == tt-1
%         TeamRankings = [TeamRankings tt];
%         TeamNameRankings = [TeamNameRankings; TeamNames{1}{tt}];
%      end
% end
% 
% end
% end
% end

SpreadChoice = Choose;
WinChoice = Choose2;
HomeOdds = OD(:,1);
AwayOdds = OD(:,2);
HomeSpread = HomeSpread';
AwaySpread = AwaySpread';
HomeTeamMargin = Prediction;

RealPicks = table(HomeSpread, HomeTeamMargin, AwaySpread, HomeOdds, AwayOdds)

Rank = [1:351]';
%table(Rank,TeamRankings',TeamNameRankings)

