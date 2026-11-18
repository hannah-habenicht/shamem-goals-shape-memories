%% Shamem 1 Behavioural: Ana02 _ Memory _ Reading in data
%%%% NR 13.05.2026
clear;
clear all;
close all;
clc;

% ---- Paths: edit these two lines ----
RAW_DIR             = 'path/to/raw_behavioural';   % raw retrieval .xlsx files (not in the data deposit)
DATA_ROOT           = 'path/to/data_deposit';      % unpacked data deposit

PATHIN              = [RAW_DIR filesep];
PATHOUT             = fullfile(DATA_ROOT, 'behavioural_results');
PATHOUT_plots       = [fullfile(PATHOUT, 'retrieval_plots') filesep];
if ~isfolder(PATHOUT), mkdir(PATHOUT); end
if ~isfolder(PATHOUT_plots), mkdir(PATHOUT_plots); end

% set parameters
number_of_subjects =100;
number_of_sessions = 2;
number_of_timing = 3; % proactive, retroactive, neutral
number_of_jobs = 3; % spy, criminal, neutral
number_of_voices = 3; % female speaker 1, female speaker 2, no speaker
detection = 0;
measures = {'dist', 'RT'};

plotting = 0;
data_retrieval = table();

% load in exclusion data
data_exclusion = readtable(fullfile(PATHOUT,'shamem_exclusion_behav.csv'));

% load accurate condition data
conditions = readmatrix(fullfile(DATA_ROOT, 'study_info', 'conditions.csv'));   % columns: id, type (timing), job, version


for s = 1:number_of_subjects
        for session = number_of_sessions
            for kind_of_timing = 1:number_of_timing
                for kind_of_job = 1:number_of_jobs
                    for kind_of_voice = 1:number_of_voices
                        file_name = [PATHIN sprintf('%03d',s) '_0' num2str(session) '_0' num2str(kind_of_timing) '_0' num2str(kind_of_job) '_0' num2str(kind_of_voice) '_shamem_study1_retrieval.xlsx'];

                        if isfile(file_name)
                          rawdata(:,:) = readmatrix(file_name);
                          session_value = session;
                        else

                        end
                    end
                end
            end
        end

        % get condition from encoding file
        timing = conditions(s,2);
        job = conditions(s,3);
        voice = conditions(s,4);
        


        if plotting == 1
            plot(1); hold on;
            scatter(rawdata(:,[7]),rawdata(:,[8]),'b');
            scatter(rawdata(:,[11]),rawdata(:,[12]),'r');
            hold off;
            figurename = ['S' num2str(s) '_retrievalplacement'];
            % save function
            filename = [PATHOUT_plots, figurename, '.svg'];
            print(gcf, '-dsvg', filename);
            close;
        end
        % order: 1 film_ID	2 coin_number	3 coin_ID	4 image	5 cond	6 onset_time
        % 7 x	8 y	9 initial_x	10 initial_y	11 response_x	12 response_y	13 L1_path	14 L2_path
        % 15 L1_precision	16 L2_precision	17 scene_time	18 coin_time	19 placement_time	20 RT_coin	21 RT_deliberation
    
        %%% chance performance for spatial distance (Data taken from rawdata information, no ID or RT
        %%% exclusion done yet.
        ChanceError = [];

        for iterations = 1 : 1000
            shuffledxy =  rawdata(randperm(length(rawdata)),11:12);
            shuffledError = sqrt((rawdata(:, 7) - shuffledxy(:, 1)).^2 + (rawdata(:, 8) - shuffledxy(:, 2)).^2);
            ChanceError(iterations) = mean(shuffledError);

        end
        ChanceErrorPP=mean(ChanceError);
        
        % subject exclusion
        data_exclusion_subject = data_exclusion(data_exclusion.id == s,:);
        
        %%% summarise results
        for trials = 1:size(rawdata,1)
            % exclusion?
            if ismember(rawdata(trials,3),data_exclusion_subject.coinID)
                exclusion = 1;
            elseif rawdata(trials,11) == 0 && rawdata(trials,12) == 0
                exclusion = 2;
            else
                exclusion = 0;
            end
            % add spatial error in Euclidean distance & RT data to table
            newrow_dist = table(s, session_value, timing, job, voice, rawdata(trials,1), rawdata(trials,3), rawdata(trials,5), trials, measures(1), rawdata(trials,16), exclusion, ...
                'VariableNames',{'id','session','timing','job', 'voice', 'filmID', 'coinID', 'category', 'trialID', 'measure', 'output','exclusion'});
            newrow_RT = table(s, session_value, timing, job, voice, rawdata(trials,1), rawdata(trials,3), rawdata(trials,5), trials, measures(2), rawdata(trials,20), exclusion, ...
                'VariableNames',{'id','session','timing','job', 'voice', 'filmID', 'coinID', 'category', 'trialID', 'measure', 'output','exclusion'});
            data_retrieval = [data_retrieval; newrow_dist; newrow_RT];
            
        end
 
    clear file_name rawdata type job voice
end

% save data output
writetable(data_retrieval, fullfile(PATHOUT,'shamem_retrieval_behav.csv'));