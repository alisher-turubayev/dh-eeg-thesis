% Automated data extraction script for BASE Mental Effort Monitoring
%   Dataset by Medeiros et al (2021)
%
% For a version with preprocessing done on data (such as bad channel 
%   removal), see `dataextract.m` script.
%
% Full reference:
%
% Medeiros, J., Couceiro, R., Duarte, G., Durães, J., Castelhano, J., 
%   Duarte, C., Castelo-Branco, M., Madeira, H., de Carvalho, P., & 
%   Teixeira, C. (2021). Can EEG Be Adopted as a Neuroscience Reference 
%   for Assessing Software Programmers' Cognitive Load? Sensors, 21(7), 
%   2338. https://doi.org/10.3390/s21072338
% 
% Work completed as part of the Master's Thesis for M.Sc. Digital Health
%   @ Hasso-Plattner Institute, University of Potsdam, Germany
% 
% Authors: Alisher Turubayev, Fabian Stolp (PhD supervisor)
% 
% Goals:
% take individual participant data in .mat format, extract required data 
%   from the original struct, import into EEGLAB and save as 
%   EEGLAB-compatible file
%
% Usage:
% 1. Change dataFolder variable to where unprocessed data from 
%       Medeiros et al. (2021) is located as needed
% 2. Change transformedDataFolder variable to change where to store 
%       processed data as needed
% 3. Change the list of participants to process - by default, the 
%       list is 27 participants
%
% Make sure the EEGLAB is in the current working directory or uncomment 
% and adjust the addpath parameter to point to EEGLAB folder - otherwise, 
% the script won't work.
% ----------------------------------------------------------------------
% IF NEEDED, uncomment this line - add EEGLAB to path
%addpath(genpath('PATH/TO/EEGLAB'))

% Define data folder and all available participants
dataFolder = '~/data/medeiros_original';
transformedDataFolder = '~/data/medeiros_raw_extracted';

% Define our participant IDs
participants = ["S01" "S03" "S04" "S05" "S07" "S08" "S09" "S10" "S11" ...
    "S12" "S13" "S14" "S16" "S17" "S18" "S19" "S20" "S21" "S22" "S23" ...
    "S24" "S25" "S26" "S27" "S28" "S29" "S30"];

% Loop over folders in the directory
%   1x4 so iterate over second dimension 
for i = 1:size(participants, 2)
    % Current folder is data path + current participant's folder path
    currFolder = append(dataFolder, '/', participants(i));

    % Find all files in the directory
    folderInfo = dir(currFolder);

    % Struct size is 5x1 so check first dimension
    if size(folderInfo, 1) == 0
        continue;
    end

    % Create the participant folder (if does not exist) and save
    %   the processed dataset there
    saveDir = append(transformedDataFolder, '/', participants(i));
    % Suppress all warnings of 'Directory already exists'
    warning('off', 'all');
    mkdir(saveDir);
    % Return back to issuing warnings
    warning('on', 'all');
    
    % Reset condition to 0
    % This variable exists to ensure that file processing goes without a
    %   hitch -> each participant is expected to have three files (three
    %   runs), thus this variable would go up to 3 for each participant
    condition = 0;

    % Iterate over each file
    for j = 1:size(folderInfo, 1)
        % If of type '.mat', start processing
        if contains(folderInfo(j).name, 'mat')
            % Increase the condition counter
            condition = condition + 1;
            disp(append('Processing file ', folderInfo(j).name));

            % Assemble a file name to load (current directory + name of 
            %   the file)
            currFilePath = append(currFolder, '/', folderInfo(j).name);

            % Assemble a file name for the event data (current directory + 
            %   name of file without extension + '_ev.txt' suffix)
            [~, name, ~] = fileparts(currFilePath);
            eventInfoFilePath = append(currFolder, '/', name, '_ev.txt');

            try 
                % Load the current file into memory 
                load(currFilePath);
            catch
                % Error reading file - simply notify and continue
                disp(append('Reading file ',name,...
                    ' failed. Continuing...'));
                continue;
            end

            % Define varialbes - unpack the originally packed data
            eegRaw = data{1,3}.data;
            eegChanInfo = data{1,3}.signals;
            eegEvents = data{2,3}.event;
            eegEventsTable = struct2table(eegEvents);
            writetable(eegEventsTable, eventInfoFilePath);
            
            % Remove used data from workspace memory
            clear data; 

            % Import into EEGLAB
            EEG = pop_importdata('dataformat','array','nbchan',68,...
                'data','eegRaw','setname',char(name),'srate',1000,...
                'subject',char(participants(i)),'pnts',0,...
                'condition',append('0', int2str(condition)),'xmin',0,...
                'chanlocs','eegChanInfo');

            % Remove used data from workspace memory
            clear eegRaw eegChanInfo eegEvents eegEvents;

            % Add event information
            EEG = pop_importevent(EEG, 'event', char(eventInfoFilePath),...
                'fields',{'type','latency','urevent'},'skipline',1,...
                'timeunit',1);

            % Run a sanity check
            EEG = eeg_checkset(EEG);

            % Save the processed dataset under the same name 
            %   (for consistency)
            EEG = pop_saveset(EEG,'filename',char(folderInfo(j).name),...
                'filepath',char(saveDir));
        end
    end
end
% Clear workspace
clear