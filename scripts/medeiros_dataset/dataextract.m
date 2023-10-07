% Automated data extraction script for BASE Mental Effort Monitoring
%   Dataset by Medeiros et al (2021)
%
% For a preprocessing-free version of this script see 
%   `dataextract_noop.m`.
%
% Full reference:
%
% Medeiros, J., Couceiro, R., Duarte, G., Durães, J., Castelhano, J., 
%   Duarte, C., Castelo-Branco, M., Madeira, H., de Carvalho, P., & 
%   Teixeira, C. (2021). Can EEG Be Adopted as a Neuroscience Reference 
%   for Assessing Software Programmers% Cognitive Load? Sensors, 21(7), 
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
%   EEGLAB-compatible file with preprocessing automated. 
%
% Usage:
% 1. Change dataFolder variable to where unprocessed data from 
%       Medeiros et al. (2021) is located as needed
% 2. Change transformedDataFolder variable to change where to store 
%       processed data as needed
% 3. Change the list of participants to process - by default, the 
%       list is 26 participants
%
% Make sure the EEGLAB is in the current working directory or uncomment 
% and adjust the addpath parameter to point to EEGLAB folder - otherwise, 
% the script won't work.
% ----------------------------------------------------------------------
% IF NEEDED, uncomment this line - add EEGLAB to path
%addpath(genpath('PATH/TO/EEGLAB'))

% IF NEEDED, adjust this line - this should point to the included
%   'locdata.ced' file
locDataPath = './thesis_scripts/locdata.ced';

% Define data folder and all available participants
dataFolder = '~/data/medeiros_original';
transformedDataFolder = '~/data/medeiros_processed_extracted';

% Define our participant IDs
participants = ["S01" "S03" "S04" "S05" "S07" "S08" "S10" "S11" "S12"...
    "S13" "S14" "S16" "S17" "S18" "S19" "S20" "S21" "S22" "S23" "S24"...
    "S25" "S26" "S27" "S28" "S29" "S30"];

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

            % Import into EEGLAB and start processing
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

            % Remove unnecessary channels -> 60 channels left
            EEG = pop_select(EEG,'rmchannel',...
                {'M1','M2','CB1','CB2','HEO','VEO','EKG','EMG'});
            % Clear the subset of them from EEG.chaninfo.removedchans -
            %   this is important for interpolation (if we do not 
            %   remove these channels, we will interpolate them too)
            EEG.chaninfo.removedchans = [];

            % Add channel location information
            EEG = pop_chanedit(EEG, 'load', {char(locDataPath)});

            % Apply filter at 1Hz / 90Hz
            EEG = pop_eegfiltnew(EEG, 'locutoff',1);
            EEG = pop_eegfiltnew(EEG, 'hicutoff',90);

            % Apply notch filter at 50 Hz
            % 
            % NOTE: Default values used
            EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',1:60,...
                'computepower',0,'linefreqs',50,'newversion',0,...
                'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,...
                'scanforlines',0,'sigtype','Channels',...
                'taperbandwidth',2,'tau',100,'verb',0,'winsize',3,...
                'winstep',1);

            % Auto-label bad channels and remove (only channels are 
            %   removed, bad sections are not - to avoid NaN values in 
            %   the final dataset)
            % 
            % NOTE: Default values are used for detecting bad channels
            %   - perhaps more aggressive values could be investigated

            EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,...
                'ChannelCriterion',0.8,'LineNoiseCriterion',4,...
                'Highpass','off','BurstCriterion','off',...
                'WindowCriterion','off','BurstRejection','off',...
                'Distance','Euclidian');
            
            % Add back bad channels by interpolation
            EEG = eeg_interp(EEG, EEG.chaninfo.removedchans, 'spherical');
            
            % Re-reference to average reference
            EEG = pop_reref(EEG, []);

            % Do a ICA decomposition and auto-label components
            EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,...
                'interrupt','off');
            EEG = pop_iclabel(EEG, 'default');

            % Flag components identified as artifacts:
            %  Muscle artifacts with 0.8 to 1 confidence
            %  Eye artifacts with 0.85 to 1 confidence
            %  Line noise with 0.85 to 1 confidence
            %  Channel noise with 0.85 to 1 confidence
            %  
            % Order of the input Brain, Muscle, Eye, Heart, Line Noise, 
            %   Channel Noise, Other
            EEG = pop_icflag(EEG,...
                [NaN NaN; 0.8 1; 0.8 1; 0.8 1; 0.8 1; 0.8 1; NaN NaN]);
            EEG = pop_subcomp(EEG, [], 0);
           
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