%function [] = Exc2Mat2()

clc
clear

numTeams = 351;

fileID = fopen('TeamNamesBB.txt','r');
TeamNames = textscan(fileID,'%s','Delimiter','\n'); % extract stats from team's txtfile
fclose(fileID);

Spread = zeros(numTeams,1);
GameN = zeros(numTeams,1);

for ii=1:numTeams
  
    %////////////////////////////////
    [ExcNum, ExcText] = xlsread('WebScrapeTest Again.xlsm',ii);
    
    Name = [TeamNames{1}{ii},'BBXS2018.mat'];
    
    save(Name, 'ExcNum', 'ExcText');
    %//////////////////////////
    
    Score = ExcNum(:,1);
    OScore = ExcNum(:,2);
    Spread(ii) = mean(Score) - mean(OScore);
    
    GameN(ii) = length(Score);
    
    clear -regexp ^ExcNum ^ExcText
    
    ii
    
end

save('QStats.mat', 'Spread', 'GameN');

%end