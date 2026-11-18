%% Shamem free recall Behavioural: Ana03 _ free recall _ Reading in data
%%%% NR 03.06.2026
clear;
close all;
clc;

% ---- Paths: edit these two lines ----
RAW_DIR             = 'path/to/data_deposit/rawdata/freerecall/MAXQDA_output';   % MAXQDA statistics per participant (unzipped from the data deposit)
DATA_ROOT           = 'path/to/data_deposit';      % unpacked data deposit

PATHIN              = RAW_DIR;
PATHOUT             = fullfile(DATA_ROOT, 'behavioural_results');
if ~isfolder(PATHOUT), mkdir(PATHOUT); end

% set parameters
number_of_subjects = 100;
session_value = 2;

% load accurate condition data
conditions = readmatrix(fullfile(DATA_ROOT, 'study_info', 'conditions.csv'));   % columns: id, type (timing), job, version


possible_categories = {'GESAMT (gültig)','not movie related','signs of insecurity',...
    'personal details','spatial details','object details','verbal speech','others',...
    'wrong personal details','wrong spatial details','wrong object details','wrong verbal speech','wrong other details'};
category_names = {'overall','unrelated','insecurity','person','spatial','object','speech','other','person','spatial','object','speech','other'};

category_totals = {'person','spatial','object','speech','other'};

measure_names = {'count', 'percent'};
data_freerecall = table();

%% get data for each participant and collate
for s = 1:number_of_subjects
    file_name = fullfile(PATHIN, sprintf('%03d_Statistik für Dokumentvariablen.xlsx', s));
    if isfile(file_name)
        % read data
        rawdata = readtable(file_name);
        rawdata.Properties.VariableNames(1:4)= {'type', 'frequency', 'percent','percent_valid'};
        
        % arrange data
        % for long format needed: subjectID, timing, job, voice, category (face, spatial, object,
        % other), right or wrong, measure type (count, percent), output
        
        % interesting variables to analyse: (1) word count overall,
        % (2) category (total, insecurity, not movierelated, face, spatial, object, verbal speech, other), (3) right or wrong, 
        for categories = 1:length(possible_categories)
            row_index = find(strcmp(rawdata{:,1},possible_categories(categories)));
            if ~isempty(row_index)
                if contains(rawdata{row_index,1},'wrong')
                    correct = {'wrong'};
                elseif contains(rawdata{row_index,1},'details') || contains(rawdata{row_index,1},'verbal') ||contains(rawdata{row_index,1},'others')
                    correct = {'right'};
                else
                    correct = {'total'};
                end
                
                % add new data rows
                newrow_freerecall = table(s, session_value, conditions(s,2),conditions(s,3), conditions(s,4), category_names(categories), correct, measure_names(1), rawdata{row_index,2}, ...
                    'VariableNames',{'id','session','timing','job', 'voice', 'category', 'correct', 'measure', 'output'});
                newrow_freerecall_percent = table(s, session_value, conditions(s,2),conditions(s,3), conditions(s,4), category_names(categories), correct, measure_names(2), rawdata{row_index,4}, ...
                    'VariableNames',{'id','session','timing','job', 'voice', 'category', 'correct', 'measure', 'output'});
                data_freerecall = [data_freerecall; newrow_freerecall; newrow_freerecall_percent];
            end
            clear correct row_index newrow_freerecall newrow_freerecall_percent
        end
        
        % add total ones (right and wrong together)
        correct = {'total'};
        for overall_categories = 1:length(category_totals)
                        newrow_freerecall = table(s, session_value, conditions(s,2),conditions(s,3), conditions(s,4), category_totals(overall_categories), correct, measure_names(1), sum(rawdata{find(contains(rawdata{:,1},category_totals{overall_categories})),2}), ...
                    'VariableNames',{'id','session','timing','job', 'voice', 'category', 'correct', 'measure', 'output'});
                newrow_freerecall_percent = table(s, session_value, conditions(s,2),conditions(s,3), conditions(s,4), category_totals(overall_categories), correct, measure_names(2), sum(rawdata{find(contains(rawdata{:,1},category_totals{overall_categories})),4}), ...
                    'VariableNames',{'id','session','timing','job', 'voice', 'category', 'correct', 'measure', 'output'});
                data_freerecall = [data_freerecall; newrow_freerecall; newrow_freerecall_percent];
        end
                
        clear correct newrow_freerecall newrow_freerecall_percent

    end
end

% save
writetable(data_freerecall, fullfile(PATHOUT,'shamem_retrieval_freerecall.csv'));

