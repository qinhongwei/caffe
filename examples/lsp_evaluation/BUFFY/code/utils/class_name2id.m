function id = class_name2id(name)
% id = class_name2id(name)
%
% object class name-to-id convertor
%
% names are case-insensitive
%
% See also class_id2name
%

switch lower(name)
    case 'ubp'
        id = 1;
    case 'ubf'
        id = 2;
    case 'full'
        id = 3;
    case 'ubfreg' %if it is from regressed ubf from face detection treat the same as normal ubf 
        id = 2;
    otherwise
        error([mfilename ': unknown class name ' name]);
end
