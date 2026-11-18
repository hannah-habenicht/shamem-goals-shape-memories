%% Shamem Study 1 Behavioural: Ana01 _ Encoding _ Reading in data
%%%% NR 13.05.2026
%
% Reference script: documents how shamem_encoding_behav.csv and
% shamem_exclusion_behav.csv were created from the raw encoding .xlsx files.
% The raw files are not public; the output files are in the Zenodo deposit,
% which is the starting point for beh_ana04.
%
% Note: the condition columns (timing, job, voice) in the output are taken from
% the raw file names. For participant 14 these differ from the corrected group
% assignment in conditions.csv (Zenodo deposit), which the analyses should use.
clear;
clear all;
close all;
clc;

% ---- Paths: edit these two lines ----
RAW_DIR             = 'path/to/raw_behavioural';   % raw encoding .xlsx files (not in the data deposit)
DATA_ROOT           = 'path/to/data_deposit';      % unpacked data deposit

PATHIN              = [RAW_DIR filesep];
PATHOUT             = fullfile(DATA_ROOT, 'behavioural_results');
if ~isfolder(PATHOUT), mkdir(PATHOUT); end

% set parameters
number_of_subjects =100;
number_of_sessions = 2;
number_of_timing = 3; % proactive, retroactive, neutral
number_of_jobs = 3; % spy, criminal, neutral
number_of_voices = 3; % female speaker 1, female speaker 2, no speaker
number_of_blocks = 3;
detection = 0;
measures = {'acc', 'RT'};
%
% timing = [];
% job = [];
% voice = [];
% rawdata = [];

% results.dimord = 'block1_face-place-object_b2_b3_types_jobs_voices';
data_encoding = table();
data_exclusion = table();
conditions = nan(number_of_subjects,4);

for s = 1:number_of_subjects
    % read in files depending on different conditions
    for session = 1:number_of_sessions
        for kind_of_type = 1:number_of_timing
            for kind_of_job = 1:number_of_jobs
                for kind_of_voice = 1:number_of_voices
                    for blocks = 1:number_of_blocks
                        file_name = [PATHIN sprintf('%03d',s) '_0' num2str(session) '_0' num2str(kind_of_type) '_0' num2str(kind_of_job) '_0' num2str(kind_of_voice) '_shamem_study1_encoding_' num2str(blocks) '.xlsx'];

                        if isfile(file_name)
                            rawdata(:,:,blocks) = readmatrix(file_name);
                            session_value = session;
                            timing = kind_of_type;
                            job = kind_of_job;
                            voice = kind_of_voice;
                        else
                            
                        end
                    end
                end
            end
        end
    end
    % order: 1 film_ID	2 coin_number 3coin_ID	4 image	5 cond	6 onset_time
    % 7 x	8 y	9 in_film_time	10 stim_presentation_time	11 click_time	12 RT

    % save conditions
    conditions(s,:) = [s, timing, job, voice];

    % get data
    for blocks_number = 1:number_of_blocks
        block_data = squeeze(rawdata(:,:,blocks_number));
        
        %%% summarise results
        for trials_block = 1:size(block_data,1)
            % detection
            if isnan(block_data(trials_block,11))
                detection = 0;
                clear newrow_RT
            else
                detection = 1;
                % if detected, define RT
                newrow_RT = table(s, session_value, timing, job, voice, blocks_number, block_data(trials_block,1), block_data(trials_block,3), block_data(trials_block,5), trials_block, measures(2), block_data(trials_block, 12), ...
                    'VariableNames',{'id','session','timing','job', 'voice', 'block', 'filmID', 'coinID', 'category', 'trialID', 'measure', 'output'});
            end
            
            % add detection & RT data to table
            newrow_acc = table(s, session_value, timing, job, voice, blocks_number, block_data(trials_block,1), block_data(trials_block,3), block_data(trials_block,5), trials_block, measures(1), detection, ...
                'VariableNames',{'id','session','timing','job', 'voice', 'block', 'filmID', 'coinID', 'category', 'trialID', 'measure', 'output'});
            if detection == 1
                data_encoding = [data_encoding; newrow_acc; newrow_RT];
            else
                data_encoding = [data_encoding; newrow_acc];
            end
            
        end
    end
    
    % is exclusion necessary for retrieval (coin not found across blocks)
    block_value = [];
    unique_coins = sort(rawdata(:,3,1));

    for coin_check = 1:size(unique_coins,1)
        block_value(1,1) = isnan(rawdata(rawdata(:,3,1) == unique_coins(coin_check),12,1));
        block_value(1,2) = isnan(rawdata(rawdata(:,3,2) == unique_coins(coin_check),12,2));
        block_value(1,3) = isnan(rawdata(rawdata(:,3,3) == unique_coins(coin_check),12,3));
        if mean(block_value)==1 %|| block_value(1,3) == 1
            newrow_excl = table(s, session_value, timing, job, voice, unique_coins(coin_check),  ...
                'VariableNames',{'id','session','timing','job', 'voice', 'coinID'});
            data_exclusion = [data_exclusion; newrow_excl];
        end
        
    end
    
    
    clear file_name rawdata type job voice
end

% save data output
writetable(data_encoding, fullfile(PATHOUT,'shamem_encoding_behav.csv'));
writetable(data_exclusion, fullfile(PATHOUT,'shamem_exclusion_behav.csv'));

% Group assignment is not saved by this script. The corrected conditions.csv
% used by the analyses is provided in the Zenodo deposit.